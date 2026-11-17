"""
Unit tests for Multi-Agent Research Assistant Backend (Google Gemini Integration).
All tests use mocks or local logic; zero external API calls are performed.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from main import (
    validate_topic,
    validate_environment,
    check_environment_status,
    resolve_gemini_api_key,
    sanitize_sensitive_text,
    build_research_crew,
    run_research,
    TopicValidationError,
    ResearchWorkflowError,
    MIN_TOPIC_LENGTH,
    MAX_TOPIC_LENGTH
)
from crewai import Process


class TestTopicValidation(unittest.TestCase):
    """Test suite for research topic sanitization and validation."""

    def test_valid_topics(self):
        valid_topics = [
            "Quantum Computing in Cancer Research",
            "Renewable energy storage microgrids 2025",
            "  AI in Healthcare  "  # Should be stripped
        ]
        for topic in valid_topics:
            cleaned = validate_topic(topic)
            self.assertEqual(cleaned, topic.strip())
            self.assertGreaterEqual(len(cleaned), MIN_TOPIC_LENGTH)

    def test_empty_and_whitespace_topic(self):
        with self.assertRaises(TopicValidationError):
            validate_topic("")
        with self.assertRaises(TopicValidationError):
            validate_topic("    \t\n  ")

    def test_too_short_topic(self):
        with self.assertRaises(TopicValidationError):
            validate_topic("AI")

    def test_too_long_topic(self):
        oversized = "A" * (MAX_TOPIC_LENGTH + 1)
        with self.assertRaises(TopicValidationError):
            validate_topic(oversized)

    def test_non_string_topic(self):
        for invalid in [None, 12345, ["topic"], {"topic": "AI"}]:
            with self.assertRaises(TopicValidationError):
                validate_topic(invalid)


class TestEnvironmentValidation(unittest.TestCase):
    """Test suite for Gemini environment variable checks and credential security."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "SERPER_API_KEY": ""}, clear=True)
    def test_missing_both_keys(self):
        status = check_environment_status()
        self.assertFalse(status["GEMINI_API_KEY"])
        self.assertFalse(status["SERPER_API_KEY"])
        with self.assertRaises(EnvironmentError) as ctx:
            validate_environment()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))
        self.assertIn("SERPER_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "your_gemini_api_key_here", "GOOGLE_API_KEY": "", "SERPER_API_KEY": "valid_key"}, clear=True)
    def test_placeholder_gemini_key(self):
        status = check_environment_status()
        self.assertFalse(status["GEMINI_API_KEY"])
        self.assertTrue(status["SERPER_API_KEY"])
        with self.assertRaises(EnvironmentError) as ctx:
            validate_environment()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "valid_gemini_key", "SERPER_API_KEY": "your_serper_api_key_here"}, clear=True)
    def test_placeholder_serper_key(self):
        status = check_environment_status()
        self.assertTrue(status["GEMINI_API_KEY"])
        self.assertFalse(status["SERPER_API_KEY"])
        with self.assertRaises(EnvironmentError) as ctx:
            validate_environment()
        self.assertIn("SERPER_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key_1", "SERPER_API_KEY": "valid_key_2"}, clear=True)
    def test_valid_keys_pass(self):
        status = check_environment_status()
        self.assertTrue(status["GEMINI_API_KEY"])
        self.assertTrue(status["SERPER_API_KEY"])
        validate_environment()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gemini_key_123", "GOOGLE_API_KEY": ""}, clear=True)
    def test_resolves_and_syncs_google_api_key(self):
        resolved = resolve_gemini_api_key()
        self.assertEqual(resolved, "gemini_key_123")
        self.assertEqual(os.environ.get("GOOGLE_API_KEY"), "gemini_key_123")


class TestSecretSanitization(unittest.TestCase):
    """Test suite for ensuring credentials are never exposed in error text or logs."""

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "AIzaSySecretGeminiKey1234567890abcdef",
        "SERPER_API_KEY": "serper-secret-api-key-99999"
    })
    def test_sanitizes_active_keys(self):
        raw_error = "Failed to connect with token AIzaSySecretGeminiKey1234567890abcdef and serper-secret-api-key-99999"
        clean = sanitize_sensitive_text(raw_error)
        self.assertNotIn("AIzaSySecretGeminiKey1234567890abcdef", clean)
        self.assertNotIn("serper-secret-api-key-99999", clean)
        self.assertIn("[GEMINI_API_KEY_REDACTED]", clean)
        self.assertIn("[SERPER_API_KEY_REDACTED]", clean)

    def test_sanitizes_generic_aizasy_pattern(self):
        raw_error = "Unauthorized error: key AIzaSyDUMMYKEYabcdef1234567890 rejected"
        clean = sanitize_sensitive_text(raw_error)
        self.assertNotIn("AIzaSyDUMMYKEYabcdef1234567890", clean)
        self.assertIn("[REDACTED_GEMINI_KEY]", clean)


class TestCrewArchitectureAndChaining(unittest.TestCase):
    """Test suite verifying agent roles, tools, task chaining, and Gemini model binding."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_gemini_key", "SERPER_API_KEY": "test_serper_key"})
    def test_crew_configuration_and_context_passing(self):
        crew = build_research_crew("Impact of Microplastics on Marine Life")

        # Verify 3 agents
        self.assertEqual(len(crew.agents), 3)
        researcher = crew.agents[0]
        writer = crew.agents[1]
        fact_checker = crew.agents[2]

        self.assertEqual(researcher.role, "Senior Research Analyst")
        self.assertEqual(writer.role, "Lead Technical & Research Writer")
        self.assertEqual(fact_checker.role, "Chief Fact Checker & Quality Assurance Auditor")

        # Verify Gemini model is assigned to all agents
        for agent in [researcher, writer, fact_checker]:
            self.assertIn("gemini-2.5-flash", agent.llm.model)
            self.assertFalse(agent.allow_delegation)

        # Verify tool assignment
        self.assertEqual(len(researcher.tools), 1, "Researcher must have SerperDevTool")
        self.assertEqual(researcher.tools[0].name, "Search the internet with Serper")
        self.assertEqual(len(writer.tools), 0, "Writer must rely strictly on research notes")
        self.assertEqual(len(fact_checker.tools), 1, "Fact Checker must have SerperDevTool")

        # Verify 3 tasks in sequential order
        self.assertEqual(len(crew.tasks), 3)
        research_task = crew.tasks[0]
        writing_task = crew.tasks[1]
        checking_task = crew.tasks[2]

        self.assertEqual(crew.process, Process.sequential)
        self.assertEqual(research_task.agent, researcher)
        self.assertEqual(writing_task.agent, writer)
        self.assertEqual(checking_task.agent, fact_checker)

        # Verify context passing dependencies
        self.assertNotIsInstance(research_task.context, list)
        self.assertEqual(writing_task.context, [research_task])
        self.assertEqual(checking_task.context, [research_task, writing_task])


class TestMockedWorkflowExecution(unittest.TestCase):
    """Test full multi-agent workflow execution using a synthetic mocked response."""

    SYNTHETIC_OUTPUT = """## PART I: Fact-Checking Verification Summary

### Audit Overview
- Total Claims Audited: 4
- [SUPPORTED]: 3
- [UNSUPPORTED]: 0
- [UNRESOLVED]: 1

### Claims Classification Table
| Claim | Source / URL | Status | Auditor Notes |
| :--- | :--- | :--- | :--- |
| Global plastic production reached 400M metric tons in 2022 | [OECD Global Plastics Outlook](https://www.oecd.org/plastics) | [SUPPORTED] | Verified against official OECD statistics. |
| Microplastics have been identified in over 1,300 marine species | [Nature Scientific Reports](https://www.nature.com/articles/s41598-020-00000-0) | [SUPPORTED] | Confirmed in peer-reviewed meta-analysis. |
| Oceanic plastic breakdown takes between 100 and 500 years | [Woods Hole Oceanographic](https://www.whoi.edu/oceanus/plastics) | [SUPPORTED] | Verified polymer degradation estimates. |
| Certain bacterial strains can eliminate 90% of oceanic plastics by 2030 | [Biotech Daily](https://example.com/unverified-claim) | [UNRESOLVED] | Lab-scale only; speculative claims cannot be confirmed for open ocean. |

---

## PART II: Corrected Final Research Report

# Research Report: Marine Microplastic Pollution and Ecosystem Impact

### 1. Executive Summary
Marine microplastics constitute a critical anthropogenic challenge impacting marine ecosystems worldwide.

### 2. Introduction
Plastic pollution in marine environments presents severe ecotoxicological risks to marine biodiversity and food safety.

### 3. Background
Since the 1950s, global synthetic polymer production has expanded exponentially, leading to pervasive microplastic fragmentation.

### 4. Key Research Findings
- Plastic output exceeded 400M tons ([OECD](https://www.oecd.org/plastics)).
- Over 1,300 species ingest microplastics ([Nature](https://www.nature.com/articles/s41598-020-00000-0)).
- Degradation intervals span 100-500 years ([Woods Hole](https://www.whoi.edu/oceanus/plastics)).

### 5. Analysis
Trophic transfer represents a primary biological concern, with microplastics accumulating across apex predators.

### 6. Limitations
Long-term toxicology studies in human populations remain sparse; bacterial bioremediation at open-sea scale remains speculative ([UNRESOLVED]).

### 7. Conclusion
Mitigation requires combined international policy interventions, reduction of virgin plastic production, and enhanced waste recovery.

### 8. References
1. OECD (2022). Global Plastics Outlook. https://www.oecd.org/plastics
2. Nature Scientific Reports (2020). Ingestion across marine taxa. https://www.nature.com/articles/s41598-020-00000-0
3. Woods Hole Oceanographic Institution. Degradation of Marine Plastics. https://www.whoi.edu/oceanus/plastics
"""

    @patch("main.validate_environment")
    @patch("main.build_research_crew")
    def test_mocked_run_research_structure(self, mock_build, mock_validate):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = self.SYNTHETIC_OUTPUT
        mock_build.return_value = mock_crew

        result = run_research("Impact of Microplastics on Marine Life")
        result_str = str(result)

        # Verify Part I Verification Summary is present
        self.assertIn("## PART I: Fact-Checking Verification Summary", result_str)
        self.assertIn("[SUPPORTED]", result_str)
        self.assertIn("[UNRESOLVED]", result_str)
        self.assertIn("Claims Classification Table", result_str)

        # Verify Part II Corrected Final Report is present
        self.assertIn("## PART II: Corrected Final Research Report", result_str)

        # Verify all 8 required report sections are present
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
        for section in required_sections:
            self.assertIn(section, result_str, f"Missing required section: {section}")

        # Verify citations and URLs
        self.assertIn("https://www.oecd.org/plastics", result_str)
        self.assertIn("https://www.nature.com", result_str)


class TestWorkflowErrorHandling(unittest.TestCase):
    """Test suite for Gemini error translation and exception security."""

    @patch("main.validate_environment")
    @patch("main.build_research_crew")
    def test_gemini_permission_denied_403(self, mock_build, mock_validate):
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("Google Gemini API error: 403 PERMISSION_DENIED. API key not valid.")
        mock_build.return_value = mock_crew

        with self.assertRaises(ResearchWorkflowError) as ctx:
            run_research("Valid research topic")
        self.assertIn("Gemini API authentication failed", str(ctx.exception))

    @patch("main.validate_environment")
    @patch("main.build_research_crew")
    def test_gemini_resource_exhausted_429(self, mock_build, mock_validate):
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("Google Gemini API error: 429 RESOURCE_EXHAUSTED. Quota exceeded.")
        mock_build.return_value = mock_crew

        with self.assertRaises(ResearchWorkflowError) as ctx:
            run_research("Valid research topic")
        self.assertIn("Gemini rate limit or quota exceeded", str(ctx.exception))

    @patch("main.validate_environment")
    @patch("main.build_research_crew")
    def test_gemini_high_demand_503(self, mock_build, mock_validate):
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("Google Gemini API error: 503 UNAVAILABLE. This model is experiencing high demand.")
        mock_build.return_value = mock_crew

        with self.assertRaises(ResearchWorkflowError) as ctx:
            run_research("Valid research topic")
        self.assertIn("Gemini model is currently experiencing high demand", str(ctx.exception))

    @patch("main.validate_environment")
    @patch("main.build_research_crew")
    def test_generic_exception_sanitizes_credentials(self, mock_build, mock_validate):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyLeakedKey1234567890abcdef"}):
            mock_crew = MagicMock()
            mock_crew.kickoff.side_effect = RuntimeError("Crash while sending to AIzaSyLeakedKey1234567890abcdef")
            mock_build.return_value = mock_crew

            with self.assertRaises(ResearchWorkflowError) as ctx:
                run_research("Valid research topic")
            err_msg = str(ctx.exception)
            self.assertNotIn("AIzaSyLeakedKey1234567890abcdef", err_msg)
            self.assertIn("[GEMINI_API_KEY_REDACTED]", err_msg)


if __name__ == "__main__":
    unittest.main()
