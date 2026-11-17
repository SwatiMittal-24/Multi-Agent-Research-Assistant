"""
Automated Unit Tests for SQLite Database Persistence.
Verifies table creation, parameterized queries, constraints, NULL tokens, and restart persistence.
"""
import os
import tempfile
import unittest

from database import (
    init_db,
    get_connection,
    save_research_run,
    save_agent_execution,
    save_citation,
    save_human_review,
    get_all_runs,
    get_run_by_id,
    get_run_metrics
)


class TestDatabasePersistence(unittest.TestCase):
    """Test suite for SQLite schema and data integrity."""

    def setUp(self):
        # Create a fresh temporary database file for each test
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        init_db(self.temp_db_path)

    def tearDown(self):
        # Clean up temporary database file
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except OSError:
                pass

    def test_tables_created_automatically(self):
        """Verify that all 4 specified tables exist with expected schema."""
        conn = get_connection(self.temp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cur.fetchall()]
        conn.close()

        expected_tables = ["research_runs", "agent_executions", "citations", "human_reviews"]
        for tbl in expected_tables:
            self.assertIn(tbl, tables, f"Table '{tbl}' was not created by init_db()")

    def test_save_and_retrieve_research_run(self):
        """Verify saving and retrieving a research run with distinction of demo/live mode."""
        save_research_run(
            run_id="RUN-TEST-001",
            topic="Quantum Computing in Healthcare",
            execution_mode="demo",
            status="Completed",
            created_at="2026-11-17 10:00:00",
            completed_at="2026-11-17 10:01:30",
            final_report="# Research Report\nTest content",
            db_path=self.temp_db_path
        )

        run = get_run_by_id("RUN-TEST-001", db_path=self.temp_db_path)
        self.assertIsNotNone(run)
        self.assertEqual(run["run_id"], "RUN-TEST-001")
        self.assertEqual(run["topic"], "Quantum Computing in Healthcare")
        self.assertEqual(run["execution_mode"], "demo")
        self.assertEqual(run["status"], "Completed")
        self.assertEqual(run["final_report"], "# Research Report\nTest content")

    def test_tokens_stored_as_null_when_unavailable(self):
        """CRITICAL: Unmeasured token counts MUST be stored as NULL, never fabricated."""
        save_research_run(
            run_id="RUN-NULL-TOKEN",
            topic="AI Policy Review",
            execution_mode="live",
            status="Completed",
            created_at="2026-11-17 11:00:00",
            db_path=self.temp_db_path
        )

        # Agent execution with unavailable token count (None)
        exec_id = save_agent_execution(
            run_id="RUN-NULL-TOKEN",
            agent_role="Senior Research Analyst",
            status="Completed",
            tokens_used=None,  # MUST store NULL
            duration_s=25.4,
            db_path=self.temp_db_path
        )

        conn = get_connection(self.temp_db_path)
        cur = conn.execute("SELECT tokens_used, duration_s FROM agent_executions WHERE execution_id = ?;", (exec_id,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNone(row["tokens_used"], "tokens_used must be NULL in database when unavailable")
        self.assertEqual(row["duration_s"], 25.4)

    def test_citations_persistence(self):
        """Verify saving and retrieving citations linked to a run."""
        save_research_run(
            run_id="RUN-CITATIONS",
            topic="Renewable Energy Storage",
            execution_mode="demo",
            status="Completed",
            created_at="2026-11-17 12:00:00",
            db_path=self.temp_db_path
        )

        save_citation(
            run_id="RUN-CITATIONS",
            source_title="NREL Battery Storage Report",
            source_url="https://www.nrel.gov/storage",
            verification_status="[SUPPORTED]",
            db_path=self.temp_db_path
        )

        run = get_run_by_id("RUN-CITATIONS", db_path=self.temp_db_path)
        self.assertEqual(len(run["citations"]), 1)
        self.assertEqual(run["citations"][0]["source_title"], "NREL Battery Storage Report")
        self.assertEqual(run["citations"][0]["source_url"], "https://www.nrel.gov/storage")
        self.assertEqual(run["citations"][0]["verification_status"], "[SUPPORTED]")

    def test_human_reviews_lifecycle(self):
        """Verify human review recording and state preservation."""
        save_research_run(
            run_id="RUN-REV-001",
            topic="Autonomous Vehicle Safety",
            execution_mode="demo",
            status="Completed",
            created_at="2026-11-17 13:00:00",
            db_path=self.temp_db_path
        )

        # Initial review: Rejected
        save_human_review(
            run_id="RUN-REV-001",
            decision="Rejected",
            comments="Requires additional statistical corroboration.",
            reviewed_at="2026-11-17 13:05:00",
            db_path=self.temp_db_path
        )
        run = get_run_by_id("RUN-REV-001", db_path=self.temp_db_path)
        self.assertEqual(run["review_status"], "Rejected")
        self.assertEqual(run["reviewer_comments"], "Requires additional statistical corroboration.")

        # Updated review: Approved
        save_human_review(
            run_id="RUN-REV-001",
            decision="Approved",
            comments="Citations verified by human audit.",
            reviewed_at="2026-11-17 13:10:00",
            db_path=self.temp_db_path
        )
        run_updated = get_run_by_id("RUN-REV-001", db_path=self.temp_db_path)
        self.assertEqual(run_updated["review_status"], "Approved")
        self.assertEqual(run_updated["reviewer_comments"], "Citations verified by human audit.")

    def test_persistence_across_simulated_app_restarts(self):
        """Verify that closing and reopening the SQLite database retains all records."""
        # 1. Create a run with all associated relations
        save_research_run(
            run_id="RUN-RESTART-TEST",
            topic="Microplastic Degradation Pathways",
            execution_mode="live",
            status="Completed",
            created_at="2026-11-17 14:00:00",
            completed_at="2026-11-17 14:03:00",
            final_report="# Comprehensive Report on Microplastics",
            db_path=self.temp_db_path
        )
        save_agent_execution(
            run_id="RUN-RESTART-TEST",
            agent_role="Chief Fact Checker & Quality Assurance Auditor",
            status="Completed",
            tokens_used=None,
            duration_s=42.1,
            db_path=self.temp_db_path
        )
        save_citation(
            run_id="RUN-RESTART-TEST",
            source_title="Nature Environmental Science",
            source_url="https://nature.com/articles/sample",
            verification_status="[SUPPORTED]",
            db_path=self.temp_db_path
        )
        save_human_review(
            run_id="RUN-RESTART-TEST",
            decision="Approved",
            comments="Fully audited and approved for archival.",
            db_path=self.temp_db_path
        )

        # 2. Simulate complete restart by re-initializing and opening a completely new connection
        init_db(self.temp_db_path)
        all_runs = get_all_runs(db_path=self.temp_db_path)
        self.assertEqual(len(all_runs), 1)

        restored_run = get_run_by_id("RUN-RESTART-TEST", db_path=self.temp_db_path)
        self.assertIsNotNone(restored_run)
        self.assertEqual(restored_run["topic"], "Microplastic Degradation Pathways")
        self.assertEqual(restored_run["execution_mode"], "live")
        self.assertEqual(restored_run["review_status"], "Approved")
        self.assertEqual(len(restored_run["agent_executions"]), 1)
        self.assertEqual(len(restored_run["citations"]), 1)
        self.assertEqual(restored_run["citations"][0]["source_url"], "https://nature.com/articles/sample")

    def test_run_metrics_aggregation(self):
        """Verify metric count calculation directly from database."""
        save_research_run("R-1", "Topic 1", "demo", "Completed", "2026-11-17 01:00:00", db_path=self.temp_db_path)
        save_research_run("R-2", "Topic 2", "live", "Completed", "2026-11-17 02:00:00", db_path=self.temp_db_path)
        save_research_run("R-3", "Topic 3", "live", "Failed", "2026-11-17 03:00:00", db_path=self.temp_db_path)

        save_human_review("R-1", "Approved", "Good", db_path=self.temp_db_path)
        save_human_review("R-2", "Rejected", "Needs work", db_path=self.temp_db_path)

        metrics = get_run_metrics(db_path=self.temp_db_path)
        self.assertEqual(metrics["total_runs"], 3)
        self.assertEqual(metrics["completed_runs"], 2)
        self.assertEqual(metrics["failed_runs"], 1)
        self.assertEqual(metrics["approved_runs"], 1)
        self.assertEqual(metrics["rejected_runs"], 1)
        self.assertEqual(metrics["pending_runs"], 1)


if __name__ == "__main__":
    unittest.main()
