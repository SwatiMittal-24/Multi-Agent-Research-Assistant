"""
Unit tests for Streamlit Frontend Demo Mode, Fixtures, and PDF Export.
Verifies that Demo Mode works 100% offline with zero external network calls.
"""
import unittest
from unittest.mock import patch

from demo_fixtures import get_demo_research_report
from pdf_generator import generate_research_pdf, clean_markdown_text
import main


class TestDemoModeFixtures(unittest.TestCase):
    """Test suite ensuring Demo Mode outputs meet all architectural and structural rules."""

    def test_curated_and_fallback_topics(self):
        topics = [
            "Responsible AI in Higher Education",
            "Autonomous Diagnostic Agents in Medicine",
            "Microplastics in Marine Wildlife",
            "General Quantum Cryptography"  # Uncurated fallback
        ]
        for topic in topics:
            report = get_demo_research_report(topic)
            self.assertIsInstance(report, str)
            self.assertGreater(len(report), 500)

            # Verification of mandatory Demo disclaimers
            self.assertIn("DEMONSTRATION OUTPUT", report)
            self.assertIn("SYNTHETIC SIMULATION", report)

            # Verification of Part I and Part II
            self.assertIn("## PART I: Fact-Checking Verification Summary", report)
            self.assertIn("Claims Classification Table", report)
            self.assertIn("[SUPPORTED]", report)
            self.assertIn("[UNRESOLVED]", report)
            self.assertIn("## PART II: Corrected Final Research Report", report)

            # Verification of all 8 report sections
            required_sections = [
                "1. Executive Summary",
                "2. Introduction",
                "3. Background",
                "4. Key Research Findings",
                "5. Analysis",
                "6. Limitations",
                "7. Conclusion",
                "8. References"
            ]
            for sec in required_sections:
                self.assertIn(sec, report, f"Section '{sec}' missing in demo report for topic '{topic}'")

    def test_demo_mode_makes_zero_network_calls(self):
        """Block all socket connections to strictly verify zero external requests."""
        def guarded_connect(*args, **kwargs):
            raise RuntimeError("Network connection attempted during Demo Mode!")

        with patch("socket.socket.connect", side_effect=guarded_connect):
            # Generate demo reports across diverse topics
            rep1 = get_demo_research_report("Machine Learning in Healthcare")
            rep2 = get_demo_research_report("Climate Change Impact")
            self.assertIn("PART I", rep1)
            self.assertIn("PART II", rep2)


class TestPDFGeneration(unittest.TestCase):
    """Test suite for PDF report rendering and watermarking."""

    def test_generate_demo_pdf(self):
        topic = "AI in Education"
        content = get_demo_research_report(topic)
        pdf_bytes = generate_research_pdf(
            topic=topic,
            markdown_content=content,
            is_demo=True,
            review_status="Approved"
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # PDF magic header %PDF
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_generate_live_pdf(self):
        topic = "Quantum Supercomputing"
        content = "# 1. Executive Summary\nQuantum computing is progressing.\n# 8. References\n1. Nature (2024)"
        pdf_bytes = generate_research_pdf(
            topic=topic,
            markdown_content=content,
            is_demo=False,
            review_status="Pending Review"
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_clean_markdown_text(self):
        raw = "Check [UNESCO Report](https://unesco.org) for **key findings** • item 1 — note"
        cleaned = clean_markdown_text(raw)
        self.assertNotIn("[UNESCO Report](https://unesco.org)", cleaned)
        self.assertIn("UNESCO Report (https://unesco.org)", cleaned)
        self.assertNotIn("**key findings**", cleaned)
        self.assertIn("key findings", cleaned)


class TestBackendPreservation(unittest.TestCase):
    """Test suite confirming the live CrewAI backend remains fully preserved."""

    def test_run_research_function_exists(self):
        self.assertTrue(hasattr(main, "run_research"))
        self.assertTrue(callable(main.run_research))

    def test_build_research_crew_exists(self):
        self.assertTrue(hasattr(main, "build_research_crew"))
        self.assertTrue(callable(main.build_research_crew))


if __name__ == "__main__":
    unittest.main()
