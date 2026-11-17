"""
Multi-Agent Research Assistant with CrewAI and Google Gemini
Project 78 | Swati Mittal | ABES Engineering College, Ghaziabad

Backend Orchestration Module:
- LLM Provider: Google Gemini (gemini-2.5-flash) via native CrewAI LLM
- Researcher Agent (SerperDevTool web search & source gathering)
- Writer Agent (8-section structured report drafting with in-text citations)
- Fact Checker Agent (claim auditing, classification, verification summary & report correction)
- Sequential Process with Task Context Chaining
- Conflict Resolution for GEMINI_API_KEY and GOOGLE_API_KEY
- Robust Credential Validation, Error Handling, and Secret Sanitization
"""
import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Securely load environment variables from .env
load_dotenv()

# Active stable Gemini Flash model ID
GEMINI_MODEL_ID = "gemini/gemini-2.5-flash"

# Recognized placeholder strings indicating unconfigured keys
PLACEHOLDER_VALUES = {
    "your_gemini_api_key_here",
    "your_openai_api_key_here",
    "your_serper_api_key_here",
    "<your_gemini_api_key>",
    "<your_serper_api_key>",
    "AIzaSy...",
    "sk-...",
    ""
}

# Research topic validation constraints
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 500


class TopicValidationError(ValueError):
    """Raised when a research topic fails validation constraints."""
    pass


class ResearchWorkflowError(Exception):
    """Raised when the multi-agent workflow encounters a runtime failure."""
    pass


def resolve_gemini_api_key() -> str:
    """
    Resolve and synchronize GEMINI_API_KEY and GOOGLE_API_KEY to prevent SDK conflicts.
    Returns the resolved valid key string, or empty string if not configured.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    # Prioritize GEMINI_API_KEY; fall back to GOOGLE_API_KEY if present
    effective_key = gemini_key if (gemini_key and gemini_key not in PLACEHOLDER_VALUES) else google_key

    if effective_key and effective_key not in PLACEHOLDER_VALUES:
        # Synchronize both environment variables so all underlying libraries find the key
        os.environ["GEMINI_API_KEY"] = effective_key
        os.environ["GOOGLE_API_KEY"] = effective_key
        return effective_key

    return ""


def check_environment_status() -> Dict[str, bool]:
    """
    Check the presence and validity of required API keys without exposing their values.

    Returns:
        Dict mapping variable names to their boolean configuration status.
    """
    gemini_key = resolve_gemini_api_key()
    serper_key = os.getenv("SERPER_API_KEY", "").strip()

    return {
        "GEMINI_API_KEY": bool(gemini_key and gemini_key not in PLACEHOLDER_VALUES),
        "SERPER_API_KEY": bool(serper_key and serper_key not in PLACEHOLDER_VALUES)
    }


def validate_environment() -> None:
    """
    Validate that all required API keys are configured before starting the agent workflow.

    Raises:
        EnvironmentError: If any required key is missing or set to a placeholder.
    """
    status = check_environment_status()
    missing = []

    if not status["GEMINI_API_KEY"]:
        missing.append("GEMINI_API_KEY (Required for Google Gemini LLM reasoning. Obtain free from: https://aistudio.google.com/apikey)")
    if not status["SERPER_API_KEY"]:
        missing.append("SERPER_API_KEY (Required for Google Web Search via Serper. Obtain from: https://serper.dev)")

    if missing:
        details = "\n".join(f"  • {m}" for m in missing)
        raise EnvironmentError(
            f"Missing required API key(s):\n{details}\n\n"
            "Action required:\n"
            "  1. Copy '.env.example' to '.env' if not already done.\n"
            "  2. Add your valid GEMINI_API_KEY and SERPER_API_KEY to '.env'.\n"
            "  3. Restart the application."
        )


def sanitize_sensitive_text(text: str) -> str:
    """
    Sanitize text to guarantee that API keys and sensitive tokens are not exposed in logs or UI.
    """
    if not text:
        return text

    sanitized = str(text)
    for env_var in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "SERPER_API_KEY", "OPENAI_API_KEY"]:
        val = os.getenv(env_var, "").strip()
        if val and len(val) >= 6:
            sanitized = sanitized.replace(val, f"[{env_var}_REDACTED]")

    # Redact common Google AI Studio key patterns (starts with AIzaSy or AIza)
    sanitized = re.sub(r"AIzaSy[a-zA-Z0-9_-]{15,}", "[REDACTED_GEMINI_KEY]", sanitized)
    # Redact common OpenAI sk- tokens if present
    sanitized = re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED_OPENAI_KEY]", sanitized)
    return sanitized


def validate_topic(topic: str) -> str:
    """
    Validate and sanitize the user-supplied research topic.

    Args:
        topic: The research query or topic string.

    Returns:
        The cleaned, sanitized topic string.

    Raises:
        TopicValidationError: If the topic is empty, non-string, too short, or too long.
    """
    if not isinstance(topic, str):
        raise TopicValidationError("Research topic must be a text string.")

    cleaned = topic.strip()
    if not cleaned:
        raise TopicValidationError("Research topic cannot be empty. Please enter a valid subject.")

    if len(cleaned) < MIN_TOPIC_LENGTH:
        raise TopicValidationError(
            f"Research topic '{cleaned}' is too brief (minimum {MIN_TOPIC_LENGTH} characters required)."
        )

    if len(cleaned) > MAX_TOPIC_LENGTH:
        raise TopicValidationError(
            f"Research topic exceeds maximum length of {MAX_TOPIC_LENGTH} characters (received {len(cleaned)})."
        )

    return cleaned


def build_research_crew(topic: str) -> Crew:
    """
    Constructs the 3-agent research workflow (Researcher -> Writer -> Fact Checker)
    powered natively by Google Gemini Flash.
    Enforces sequential task execution and strict context passing between agents.

    Args:
        topic: The validated research topic.

    Returns:
        Crew: Configured CrewAI Crew instance ready for kickoff.
    """
    validate_environment()
    cleaned_topic = validate_topic(topic)
    gemini_key = resolve_gemini_api_key()

    # Native CrewAI LLM configuration using Google Gemini Flash
    # temperature=0.2 encourages factual precision and reduces hallucination risk
    llm = LLM(
        model=GEMINI_MODEL_ID,
        api_key=gemini_key,
        temperature=0.2
    )

    search_tool = SerperDevTool()

    # Agent 1: Senior Research Analyst
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Conduct thorough, evidence-based web research on '{cleaned_topic}'. Locate credible primary sources, verify facts, and compile a structured research dossier with exact URLs and dates. Never fabricate sources.",
        backstory=(
            "You are an expert investigative research analyst dedicated to absolute factual accuracy. "
            "You search the web systematically to uncover authoritative sources, academic papers, and official reports. "
            "You meticulously record source titles, canonical URLs, publication dates, and key empirical findings. "
            "You explicitly note any gaps, conflicting evidence, or uncertainties in the data. "
            "CRITICAL: You NEVER invent a citation, author, statistic, or URL."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_retry_limit=1
    )

    # Agent 2: Lead Technical & Research Writer
    writer = Agent(
        role="Lead Technical & Research Writer",
        goal=(
            f"Synthesize the research notes on '{cleaned_topic}' into an authoritative, publication-quality "
            "8-section research report. Base all content strictly on cited evidence and do not introduce ungrounded claims."
        ),
        backstory=(
            "You are a professional science and policy writer skilled at transforming raw research data into clear, balanced, structured reports. "
            "You adhere strictly to academic integrity: every factual claim is tethered to its cited source from the research notes. "
            "You organize findings methodically into 8 required sections and clearly distinguish established facts from emerging hypotheses or limitations. "
            "You NEVER introduce claims unsupported by the research evidence."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_retry_limit=1
    )

    # Agent 3: Chief Fact Checker & Quality Assurance Auditor
    fact_checker = Agent(
        role="Chief Fact Checker & Quality Assurance Auditor",
        goal=(
            f"Independently audit all factual claims, statistics, and citations in the research report on '{cleaned_topic}'. "
            "Classify every claim as [SUPPORTED], [UNSUPPORTED], or [UNRESOLVED], and deliver a verified final report with an explicit Verification Summary."
        ),
        backstory=(
            "You are a rigorous, skeptical fact-checking editor trained to hold research publications to the highest standards of veracity. "
            "You audit statements claim-by-claim, verify that URLs are real and credible, and cross-reference statistics. "
            "You use web search tools when necessary to resolve ambiguous or contested assertions. "
            "You NEVER accept unverified claims as true and never invent citations."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_retry_limit=1
    )

    # Task 1: Research and Source Gathering
    research_task = Task(
        description=(
            f"Execute in-depth web research on the topic: '{cleaned_topic}'.\n\n"
            "Requirements:\n"
            "1. Conduct targeted searches using the search tool to find at least 3 to 5 credible, authoritative sources.\n"
            "2. For each source, record:\n"
            "   - Full Source Title\n"
            "   - Canonical URL\n"
            "   - Publication Date (or latest update date, if available)\n"
            "   - Key empirical findings and supporting evidence\n"
            "3. Extract specific statistics, dates, and direct factual findings with their corresponding sources.\n"
            "4. Identify any unresolved questions, data limitations, or conflicting perspectives in the evidence.\n"
            "5. CRITICAL: Strictly forbid inventing or guessing URLs or citations. Only include sources discovered through actual search results."
        ),
        expected_output=(
            "A structured research dossier in Markdown format containing:\n"
            "1. Research Overview\n"
            "2. Detailed Findings with direct source attribution\n"
            "3. Source Reference Table (columns: #, Source Title, Canonical URL, Publication Date, Relevance)\n"
            "4. Identified Uncertainties, Data Gaps, and Conflicting Points"
        ),
        agent=researcher
    )

    # Task 2: Structured Report Writing
    writing_task = Task(
        description=(
            f"Draft a comprehensive, formal research report on '{cleaned_topic}' using EXCLUSIVELY the provided research notes.\n\n"
            "The report MUST strictly follow this exact 8-section structure:\n"
            "1. Executive Summary: High-level overview of findings and practical implications.\n"
            "2. Introduction: Scope, significance, and core questions addressed.\n"
            "3. Background: Historical context, foundational concepts, and terminology.\n"
            "4. Key Research Findings: Detailed findings organized by theme, citing specific sources.\n"
            "5. Analysis: Synthesized insights, patterns, and critical evaluation of evidence.\n"
            "6. Limitations: Methodological constraints, data gaps, or uncertainties identified in the research.\n"
            "7. Conclusion: Summary takeaways and future outlook.\n"
            "8. References: Formatted list of all sources with Titles, Canonical URLs, and dates.\n\n"
            "Citation Rules:\n"
            "- Anchor every significant factual assertion to an in-text reference citation (e.g., [Source Title](URL)).\n"
            "- Do not introduce any unsupported claims, invented numbers, or facts not present in the research notes."
        ),
        expected_output=(
            "A complete, publication-grade 8-section research report formatted in clean Markdown, "
            "with all 8 numbered sections present, in-text citations throughout, and a comprehensive References section."
        ),
        agent=writer,
        context=[research_task]
    )

    # Task 3: Fact Checking, Verification, and Report Correction
    checking_task = Task(
        description=(
            f"Perform an exhaustive line-by-line audit of the draft research report on '{cleaned_topic}'.\n\n"
            "Audit Workflow:\n"
            "1. Identify every empirical claim, statistic, date, and attribution in the draft report.\n"
            "2. Cross-verify each claim against the research notes and verify that URLs are valid. Use the search tool to verify disputed or ambiguous statements.\n"
            "3. Classify each audited claim into exactly one of three categories:\n"
            "   - [SUPPORTED]: Verified by credible cited sources.\n"
            "   - [UNSUPPORTED]: Lacks evidence, contradicts sources, or misrepresents data.\n"
            "   - [UNRESOLVED]: Inconclusive evidence, contradictory sources, or unverified claims.\n"
            "4. Compile a structured 'Fact-Checking Verification Summary' table with columns:\n"
            "   | Claim | Source / URL | Status ([SUPPORTED]/[UNSUPPORTED]/[UNRESOLVED]) | Auditor Notes |\n"
            "5. Produce a corrected final version of the 8-section research report:\n"
            "   - Correct or remove unsupported claims.\n"
            "   - For unresolved claims, attach explicit cautionary notes so they are never presented as verified facts.\n"
            "   - Preserve the complete 8-section structure.\n\n"
            "Final Deliverable Format:\n"
            "The final output must be structured into two clear parts:\n"
            "## PART I: Fact-Checking Verification Summary\n"
            "(Audit statistics and complete claims classification table)\n\n"
            "## PART II: Corrected Final Research Report\n"
            "(Complete 8-section verified report with all citations and explicit unresolved claim warnings)"
        ),
        expected_output=(
            "A comprehensive audit dossier in Markdown containing Part I (Fact-Checking Verification Summary with "
            "explicit [SUPPORTED], [UNSUPPORTED], and [UNRESOLVED] classifications) followed by Part II (the corrected, "
            "fully cited 8-section research report with clear cautionary notes for any unresolved claims)."
        ),
        agent=fact_checker,
        context=[research_task, writing_task]
    )

    return Crew(
        agents=[researcher, writer, fact_checker],
        tasks=[research_task, writing_task, checking_task],
        process=Process.sequential,
        verbose=True,
        max_rpm=10
    )


def run_research(topic: str) -> Any:
    """
    Executes the multi-agent research workflow with input validation, credential verification,
    and Gemini-specific error handling.

    Args:
        topic: The research topic string.

    Returns:
        The CrewOutput result from the CrewAI workflow.

    Raises:
        TopicValidationError: If the topic input is invalid.
        EnvironmentError: If required credentials are missing.
        ResearchWorkflowError: If workflow execution encounters a model, tool, network, or runtime failure.
    """
    cleaned_topic = validate_topic(topic)
    validate_environment()

    try:
        crew = build_research_crew(cleaned_topic)
        result = crew.kickoff()
        return result
    except EnvironmentError:
        raise
    except TopicValidationError:
        raise
    except Exception as exc:
        raw_msg = str(exc)
        safe_msg = sanitize_sensitive_text(raw_msg)

        # Gemini-specific error diagnostics
        if "403" in raw_msg or "PERMISSION_DENIED" in raw_msg or "API_KEY_INVALID" in raw_msg:
            raise ResearchWorkflowError(
                "Gemini API authentication failed (403 Permission Denied). "
                "Please verify that your GEMINI_API_KEY is active and valid in your .env file."
            ) from None
        elif "429" in raw_msg or "RESOURCE_EXHAUSTED" in raw_msg or "quota" in raw_msg.lower():
            raise ResearchWorkflowError(
                "Gemini rate limit or quota exceeded (429 Resource Exhausted). "
                "Please wait a few moments or check your Google AI Studio quota."
            ) from None
        elif "503" in raw_msg or "UNAVAILABLE" in raw_msg or "high demand" in raw_msg.lower():
            raise ResearchWorkflowError(
                "Gemini model is currently experiencing high demand (503 Service Unavailable). "
                "Please wait 30 seconds and try again."
            ) from None
        elif "404" in raw_msg or "NOT_FOUND" in raw_msg:
            raise ResearchWorkflowError(
                f"Gemini model ({GEMINI_MODEL_ID}) was not found or is unavailable for this API version (404)."
            ) from None
        else:
            raise ResearchWorkflowError(f"Research workflow encountered an error: {safe_msg}") from None
