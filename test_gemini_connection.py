"""
Temporary verification script: test_gemini_connection.py
Makes exactly one minimal Gemini API request using the configured GEMINI_API_KEY.
Does not invoke any CrewAI agents.
Never displays the API key.
"""
import os
import re
from dotenv import load_dotenv
from crewai import LLM

# Load environment credentials securely from .env
load_dotenv()

MODEL_NAME = "gemini/gemini-2.5-flash"

def test_connection():
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print(f"Model: {MODEL_NAME}")
        print("Success: False")
        print("Error: Missing or unconfigured GEMINI_API_KEY in .env")
        return False

    # Synchronize environment variable for underlying Google GenAI SDK
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["GOOGLE_API_KEY"] = gemini_key

    try:
        # Native CrewAI LLM direct call (no agents initialized)
        llm = LLM(model=MODEL_NAME, api_key=gemini_key, temperature=0.0)
        response = llm.call(messages=[{"role": "user", "content": "Reply with the word READY."}])
        
        # Format response cleanly and ensure no keys leaked
        response_text = str(response).strip()
        if gemini_key in response_text:
            response_text = response_text.replace(gemini_key, "[REDACTED]")

        print(f"Model: {MODEL_NAME}")
        print("Success: True")
        print(f"Response: {response_text}")
        return True

    except Exception as exc:
        err_str = str(exc)
        safe_err = err_str.replace(gemini_key, "[REDACTED]")
        safe_err = re.sub(r"AIzaSy[a-zA-Z0-9_-]{15,}", "[REDACTED]", safe_err)

        # Extract concise sanitized error code/summary
        if "403" in safe_err or "PERMISSION_DENIED" in safe_err:
            concise_err = "403 PERMISSION_DENIED (Invalid API key or API not enabled in Google Cloud)"
        elif "429" in safe_err or "RESOURCE_EXHAUSTED" in safe_err:
            concise_err = "429 RESOURCE_EXHAUSTED (Rate limit or quota exceeded)"
        elif "503" in safe_err or "UNAVAILABLE" in safe_err:
            concise_err = "503 UNAVAILABLE (Model high demand or service temporarily unavailable)"
        elif "404" in safe_err:
            concise_err = "404 NOT_FOUND (Model endpoint not found)"
        else:
            concise_err = safe_err[:200]

        print(f"Model: {MODEL_NAME}")
        print("Success: False")
        print(f"Error: {concise_err}")
        return False

if __name__ == "__main__":
    test_connection()
