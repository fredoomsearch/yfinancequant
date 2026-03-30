from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from assistant.scorecard import build_assistant_scorecard


class AssistantScorecardTest(unittest.TestCase):
    def test_scorecard_reports_layers_and_summary_scores(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            scorecard = build_assistant_scorecard().to_dict()

        self.assertIn("generated_at", scorecard)
        self.assertGreaterEqual(scorecard["local_first_runtime_ready_score_pct"], 90)
        self.assertGreaterEqual(scorecard["local_first_implementation_score_pct"], 90)
        self.assertGreaterEqual(scorecard["hybrid_implementation_score_pct"], 90)
        self.assertEqual(scorecard["hybrid_runtime_ready_score_pct"], scorecard["hybrid_implementation_score_pct"])
        self.assertTrue(scorecard["runtime"]["web_retriever_configured"])
        self.assertTrue(scorecard["runtime"]["web_retriever_config_valid"])
        self.assertTrue(scorecard["runtime"]["web_retriever_runtime_ready"])
        self.assertTrue(scorecard["runtime"]["web_cache_enabled"])
        self.assertEqual(scorecard["runtime"]["web_cache_ttl_seconds"], 300)
        self.assertFalse(scorecard["runtime"]["web_config_issues"])
        self.assertIn("semantic_layer", scorecard["layers"])
        self.assertIn("primary_conversational_interpreter", scorecard["layers"])
        self.assertIn("memory_reference_layer", scorecard["layers"])
        self.assertIn("web_retrieval_layer", scorecard["layers"])
        self.assertGreaterEqual(scorecard["layers"]["memory_reference_layer"]["runtime_ready_score_pct"], 90)
        self.assertEqual(scorecard["hybrid_runtime_ready_score_pct"], 100)
        self.assertEqual(scorecard["layers"]["web_retrieval_layer"]["runtime_ready_score_pct"], 100)
        self.assertEqual(scorecard["layers"]["web_retrieval_layer"]["implementation_score_pct"], 100)
        self.assertEqual(scorecard["layers"]["web_fact_extraction"]["runtime_ready_score_pct"], 100)
        self.assertEqual(scorecard["layers"]["mixed_grounding_engine"]["runtime_ready_score_pct"], 100)
        self.assertEqual(scorecard["layers"]["semantic_layer"]["runtime_ready_score_pct"], 100)

    def test_scorecard_reflects_runtime_web_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
                "ASSISTANT_WEB_SEARCH_METHOD": "GET",
                "ASSISTANT_WEB_RESULTS_PATH": "payload.hits",
            },
            clear=True,
        ):
            scorecard = build_assistant_scorecard().to_dict()

        self.assertTrue(scorecard["runtime"]["web_retriever_configured"])
        self.assertTrue(scorecard["runtime"]["web_retriever_config_valid"])
        self.assertTrue(scorecard["runtime"]["web_retriever_runtime_ready"])
        self.assertTrue(scorecard["runtime"]["web_cache_enabled"])
        self.assertEqual(scorecard["runtime"]["web_method"], "GET")
        self.assertEqual(scorecard["runtime"]["web_results_path"], "payload.hits")
        self.assertEqual(
            scorecard["layers"]["web_retrieval_layer"]["runtime_ready_score_pct"],
            scorecard["layers"]["web_retrieval_layer"]["implementation_score_pct"],
        )


if __name__ == "__main__":
    unittest.main()
