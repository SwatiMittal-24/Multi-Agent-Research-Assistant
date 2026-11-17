"""
Unit Tests for Analytics Engine and Benchmark Dataset Calculations.
Verifies KPI computations, interactive filtering, chart generation, and dataset integrity.
"""
import unittest
import pandas as pd
import plotly.graph_objects as go

from analytics_engine import (
    load_dataset,
    load_data_dictionary,
    compute_overview_kpis,
    filter_dataset,
    plot_token_usage_by_agent,
    plot_duration_by_agent,
    plot_quality_by_topic,
    plot_error_rate_by_agent,
    plot_token_distribution
)


class TestAnalyticsEngine(unittest.TestCase):
    """Test suite for dataset analytics and visualizations."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.data_dict = load_data_dictionary()

    def test_dataset_shape_and_columns(self):
        """Verify the 5,000-row dataset has exactly the 7 specified columns."""
        self.assertEqual(len(self.df), 5000)
        expected_cols = ["run_id", "agent_role", "task_type", "tokens_used", "duration_s", "quality_score", "errors"]
        self.assertListEqual(list(self.df.columns), expected_cols)

    def test_kpis_calculation(self):
        """Verify statistical aggregations match expected dataset summary."""
        kpis = compute_overview_kpis(self.df)
        self.assertEqual(kpis["total_records"], 5000)
        self.assertEqual(kpis["total_topics"], 20)
        self.assertEqual(len(kpis["agent_roles"]), 3)
        self.assertAlmostEqual(kpis["avg_tokens"], 3650.11, places=1)
        self.assertAlmostEqual(kpis["avg_duration_s"], 80.02, places=1)
        self.assertAlmostEqual(kpis["avg_quality_score"], 0.820, places=2)
        self.assertEqual(kpis["total_errors"], 147)
        self.assertAlmostEqual(kpis["error_rate_pct"], 2.94, places=2)

    def test_filter_by_agent_role(self):
        """Verify filtering by agent role returns exact subsets."""
        filtered_researcher = filter_dataset(self.df, roles=["Researcher"])
        self.assertTrue((filtered_researcher["agent_role"] == "Researcher").all())
        self.assertGreater(len(filtered_researcher), 0)
        self.assertLess(len(filtered_researcher), 5000)

        filtered_multi = filter_dataset(self.df, roles=["Writer", "Fact_Checker"])
        self.assertTrue(set(filtered_multi["agent_role"].unique()).issubset({"Writer", "Fact_Checker"}))

    def test_filter_by_topic(self):
        """Verify filtering by task type / topic."""
        filtered_topic = filter_dataset(self.df, topics=["AI_Trends", "HealthAI"])
        self.assertTrue(set(filtered_topic["task_type"].unique()).issubset({"AI_Trends", "HealthAI"}))

    def test_filter_by_error_status(self):
        """Verify filtering by error status (0 vs 1)."""
        errors_only = filter_dataset(self.df, error_status="Errors Only (1)")
        self.assertTrue((errors_only["errors"] == 1).all())
        self.assertEqual(len(errors_only), 147)

        clean_only = filter_dataset(self.df, error_status="Error-Free Only (0)")
        self.assertTrue((clean_only["errors"] == 0).all())
        self.assertEqual(len(clean_only), 5000 - 147)

    def test_visualizations_return_valid_figures(self):
        """Verify all 5 required chart functions produce valid plotly figures."""
        fig_tokens = plot_token_usage_by_agent(self.df)
        self.assertIsInstance(fig_tokens, go.Figure)

        fig_duration = plot_duration_by_agent(self.df)
        self.assertIsInstance(fig_duration, go.Figure)

        fig_quality = plot_quality_by_topic(self.df)
        self.assertIsInstance(fig_quality, go.Figure)

        fig_error = plot_error_rate_by_agent(self.df)
        self.assertIsInstance(fig_error, go.Figure)

        fig_hist = plot_token_distribution(self.df)
        self.assertIsInstance(fig_hist, go.Figure)

    def test_data_dictionary_loaded(self):
        """Verify data dictionary has all 7 documented columns."""
        self.assertEqual(len(self.data_dict), 7)
        self.assertIn("column_name", self.data_dict.columns)
        self.assertIn("description", self.data_dict.columns)

    def test_dataset_remains_unmodified(self):
        """Verify original dataset on disk has not been mutated."""
        fresh_read = pd.read_csv("dataset.csv")
        self.assertEqual(len(fresh_read), 5000)
        self.assertEqual(len(fresh_read.columns), 7)


if __name__ == "__main__":
    unittest.main()
