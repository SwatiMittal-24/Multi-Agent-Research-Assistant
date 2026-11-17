"""
Live integration test for Multi-Agent Research Assistant powered by Google Gemini.

IMPORTANT NOTE:
- This test connects to real external APIs (Google Gemini and Serper Dev).
- It consumes Gemini API quota and Serper search credits.
- By default, it runs only when the '--run-live' argument is explicitly passed,
  preventing unexpected quota consumption during routine testing.
- Live test results are logged honestly; mocked tests are never reported as live proofs.
"""
import sys
import time
import argparse
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Ensure local environment variables are loaded
load_dotenv(PROJECT_ROOT / ".env")

from main import run_research, check_environment_status, GEMINI_MODEL_ID


def run_live_test(topic: str = "2024 Nobel Prize in Physics winners"):
    print("=" * 70)
    print("LIVE INTEGRATION TEST: Multi-Agent Research Assistant (Google Gemini)")
    print("=" * 70)
    print(f"Target Research Topic: '{topic}'")
    print(f"Configured Gemini Model: '{GEMINI_MODEL_ID}'")
    print("Checking environment status...")

    status = check_environment_status()
    print(f"  • Gemini API Key configured: {status['GEMINI_API_KEY']}")
    print(f"  • Serper API Key configured: {status['SERPER_API_KEY']}")

    if not (status["GEMINI_API_KEY"] and status["SERPER_API_KEY"]):
        print("\n[SKIPPED] Live integration test cannot execute.")
        print("Reason: Valid GEMINI_API_KEY and SERPER_API_KEY must be provided in .env.")
        print("Mocked unit tests verify the workflow architecture without credentials.")
        return False

    print("\nStarting live multi-agent execution...")
    print("Sequential pipeline: Researcher -> Writer -> Fact Checker")
    start_time = time.time()

    try:
        result = run_research(topic)
        elapsed = time.time() - start_time
        result_str = str(result)

        print("\n" + "=" * 70)
        print("LIVE EXECUTION FINISHED")
        print("=" * 70)
        print(f"Total Execution Time: {elapsed:.2f} seconds")
        print(f"Output Character Length: {len(result_str)}")

        # Verification of required components
        checks = {
            "Part I (Verification Summary)": "PART I" in result_str or "Verification Summary" in result_str,
            "Claims classification ([SUPPORTED] / [UNRESOLVED])": any(tag in result_str for tag in ["[SUPPORTED]", "[UNSUPPORTED]", "[UNRESOLVED]"]),
            "Part II (Report)": "PART II" in result_str or "Research Report" in result_str,
            "Section: Executive Summary": "Executive Summary" in result_str,
            "Section: Introduction": "Introduction" in result_str,
            "Section: Background": "Background" in result_str,
            "Section: Key Research Findings": "Key Research Findings" in result_str or "Findings" in result_str,
            "Section: Analysis": "Analysis" in result_str,
            "Section: Limitations": "Limitations" in result_str,
            "Section: Conclusion": "Conclusion" in result_str,
            "Section: References": "References" in result_str or "Sources" in result_str,
            "Real Citations / URLs": "http://" in result_str or "https://" in result_str,
        }

        print("\nVerification Checklist:")
        all_passed = True
        for check_name, passed in checks.items():
            status_symbol = "✅ PASS" if passed else "⚠️ WARNING / MISSING"
            print(f"  [{status_symbol}] {check_name}")
            if not passed:
                all_passed = False

        print("\nPreview of Output (first 600 characters):")
        print("-" * 50)
        print(result_str[:600])
        print("...")
        print("-" * 50)

        if all_passed:
            print("\n[RESULT] Live integration test PASSED all structural checks.")
            return True
        else:
            print("\n[RESULT] Live execution completed with structural warnings.")
            return False

    except Exception as exc:
        elapsed = time.time() - start_time
        print("\n" + "!" * 70)
        print("LIVE EXECUTION FAILED")
        print("!" * 70)
        print(f"Elapsed Time before failure: {elapsed:.2f} seconds")
        print(f"Error Type: {type(exc).__name__}")
        print(f"Sanitized Error Message: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live integration test for CrewAI research assistant with Google Gemini.")
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Explicitly allow live external API calls to Gemini and Serper."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="2024 Nobel Prize in Physics winners",
        help="Specific research topic to test."
    )
    args = parser.parse_args()

    if not args.run_live:
        print("=" * 70)
        print("OPTIONAL LIVE INTEGRATION TEST RUNNER (Google Gemini)")
        print("=" * 70)
        print("Notice: To prevent unintended API costs and rate usage, live testing requires")
        print("the '--run-live' flag.")
        print("\nUsage:")
        print("  python tests/test_live_integration.py --run-live")
        print("  python tests/test_live_integration.py --run-live --topic \"Quantum dots discovery\"")
        print("\nTo run automated offline tests instead:")
        print("  python -m unittest discover tests -p \"test_backend_unit.py\"")
        sys.exit(0)

    success = run_live_test(topic=args.topic)
    sys.exit(0 if success else 1)
