"""服务器预检必须拒绝看似已配置、实际仍为模板的外部 job。"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_paper_setup.py"
SPEC = importlib.util.spec_from_file_location("validate_paper_setup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ValidateSetupTests(unittest.TestCase):
    def test_placeholder_method_job_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "external"
            repository.mkdir()
            config = {
                "_source": str(root / "evaluation_matrix.json"),
                "provider": {
                    "id": "relay",
                    "base_url_env": "TEST_BASE_URL",
                    "base_url_default": "https://example.test/v1",
                    "key_env": "TEST_API_KEY",
                },
                "models": [],
                "benchmarks": [
                    {
                        "id": "external-test",
                        "repository": "external",
                        "methods": ["team-memory"],
                        "method_jobs": {
                            "team-memory": {
                                "command": ["python", "path/to/adapter.py"]
                            }
                        },
                    }
                ],
            }
            with patch.dict(
                os.environ,
                {"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://example.test/v1"},
                clear=False,
            ):
                report = MODULE.inspect_setup(
                    config,
                    benchmark_ids={"external-test"},
                    require_jobs=True,
                )

        self.assertFalse(report["ok"])
        self.assertTrue(report["benchmarks"][0]["placeholder_method_jobs"])
        self.assertIn("placeholder values", report["errors"][0])

    def test_require_jobs_rejects_missing_case_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "external"
            repository.mkdir()
            config = {
                "_source": str(root / "evaluation_matrix.json"),
                "provider": {
                    "id": "relay",
                    "base_url_env": "TEST_BASE_URL",
                    "base_url_default": "https://example.test/v1",
                    "key_env": "TEST_API_KEY",
                },
                "models": [],
                "benchmarks": [
                    {
                        "id": "external-test",
                        "repository": "external",
                        "tasks": ["task-a"],
                        "methods": ["team-memory"],
                        "paper_run_methods": ["team-memory"],
                        "method_jobs": {
                            "team-memory": {"command": ["python", "adapter.py"]}
                        },
                    }
                ],
            }
            with patch.dict(
                os.environ,
                {"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://example.test/v1"},
                clear=False,
            ):
                report = MODULE.inspect_setup(
                    config,
                    benchmark_ids={"external-test"},
                    require_jobs=True,
                )

        self.assertFalse(report["ok"])
        self.assertTrue(
            any("missing case_ids" in error for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()
