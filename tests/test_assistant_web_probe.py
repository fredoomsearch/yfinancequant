from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import assistant_web_probe


class AssistantWebProbeScriptTest(unittest.TestCase):
    def test_status_only_json_accepts_provider_overrides(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = assistant_web_probe.main(
                    [
                        "--provider",
                        "tavily",
                        "--search-url",
                        "https://search.test",
                        "--api-key",
                        "secret-key",
                        "--status-only",
                        "--json",
                    ]
                )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["config_valid"])
        self.assertTrue(payload["runtime_ready"])
        self.assertEqual(payload["provider"], "tavily")
        self.assertEqual(payload["auth_param"], "api_key")
        self.assertEqual(payload["method"], "POST")

    def test_print_env_emits_provider_snippet(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = assistant_web_probe.main(
                    [
                        "--provider",
                        "serper",
                        "--search-url",
                        "https://google.serper.dev/search",
                        "--print-env",
                    ]
                )

        rendered = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn('ASSISTANT_WEB_PROVIDER="serper"', rendered)
        self.assertIn('ASSISTANT_WEB_SEARCH_URL="https://google.serper.dev/search"', rendered)
        self.assertIn('ASSISTANT_WEB_SEARCH_API_KEY="..."', rendered)

    def test_save_report_writes_probe_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "probe.json"
            with patch.dict(os.environ, {}, clear=True):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = assistant_web_probe.main(
                        [
                            "--provider",
                            "tavily",
                            "--search-url",
                            "https://search.test",
                            "--api-key",
                            "secret-key",
                            "--status-only",
                            "--save-report",
                            str(report_path),
                            "--json",
                        ]
                    )

            payload = json.loads(report_path.read_text())
            self.assertEqual(exit_code, 0)
            self.assertTrue(report_path.exists())
            self.assertTrue(payload["runtime_ready"])
            self.assertEqual(payload["provider"], "tavily")


if __name__ == "__main__":
    unittest.main()
