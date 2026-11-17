# Multi-Agent Research Assistant with CrewAI and Google Gemini

**Learner:** Swati Mittal  
**Roll No.:** 2300320100260  
**Campus:** ABES Engineering College, Ghaziabad  
**Project No.:** 78

## Architecture & Workflow
The multi-agent research system uses a sequential CrewAI workflow powered by **Google Gemini** (`gemini-2.5-flash`):
1. **Researcher Agent** (`Senior Research Analyst`): Gathers factual data, source titles, dates, and canonical URLs via Serper Google Search.
2. **Writer Agent** (`Lead Technical & Research Writer`): Synthesizes research notes into a formal 8-section research report with in-text citations.
3. **Fact Checker Agent** (`Chief Fact Checker & Quality Assurance Auditor`): Audits factual claims, classifies claims (`[SUPPORTED]`, `[UNSUPPORTED]`, `[UNRESOLVED]`), produces a Verification Summary, and provides a corrected final report.
4. **Human Review**: Mandatory human review before report publication.

## Setup
1. Install Python 3.10+.
2. Create and activate a virtual environment (`python -m venv .venv`).
3. Run `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and configure your credentials:
   - `GEMINI_API_KEY`: Obtain free from [Google AI Studio](https://aistudio.google.com/apikey).
   - `SERPER_API_KEY`: Obtain free tier from [Serper Dev](https://serper.dev).
5. Run tests: `python -m unittest discover tests -p "test_backend_unit.py"`.
6. Run the application: `streamlit run app.py`.

No API keys are committed in this repository. All credentials are securely managed through `.env` and `python-dotenv`.

## Dataset
`dataset.csv` is a fully synthetic 5,000-row demonstration dataset generated with random seed 78. The original dataset.csv was not provided in the uploaded materials. Synthetic records must not be represented as real production logs.

## Schema note
The generation prompt and uploaded data dictionary specify seven fields:
`run_id, agent_role, task_type, tokens_used, duration_s, quality_score, errors`.
Page 2 of the official Project Brief instead lists eight different fields:
`run_id, agent_role, task_description, tool_used, input_tokens, output_tokens, duration_s, quality_score`.
The primary dataset follows the prompt and data dictionary. Confirm the required schema with your mentor before final submission.
