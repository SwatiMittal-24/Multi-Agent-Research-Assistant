"""
Multi-Agent Research Assistant with CrewAI, Google Gemini, and SQLite Persistence
Streamlit Application with Safe Demo Mode and Persistent Database Storage

Learner: Swati Mittal | Roll No: 2300320100260 | ABES Engineering College | Project 78
"""
import time
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure local environment variables are loaded
load_dotenv()

# Project backend and database modules
from main import (
    run_research,
    check_environment_status,
    validate_topic,
    validate_environment,
    TopicValidationError,
    ResearchWorkflowError,
    GEMINI_MODEL_ID
)
from demo_fixtures import get_demo_research_report
from pdf_generator import generate_research_pdf
from database import (
    init_db,
    save_research_run,
    save_agent_execution,
    extract_and_save_citations_from_report,
    save_human_review,
    get_all_runs,
    get_run_by_id,
    get_run_metrics,
    get_connection
)
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

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema automatically on first launch
init_db()

# Seed default initial demo run if database is completely empty
existing_runs = get_all_runs()
if not existing_runs:
    initial_topic = "Responsible Use of Generative AI in Higher Education"
    initial_report = get_demo_research_report(initial_topic)
    initial_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    save_research_run(
        run_id="RUN-001",
        topic=initial_topic,
        execution_mode="demo",
        status="Completed",
        created_at=initial_created,
        completed_at=initial_created,
        final_report=initial_report
    )
    # Log 3 synthetic agent executions with NULL tokens
    save_agent_execution("RUN-001", "Senior Research Analyst", "Completed", tokens_used=None, duration_s=1.2)
    save_agent_execution("RUN-001", "Lead Technical & Research Writer", "Completed", tokens_used=None, duration_s=1.0)
    save_agent_execution("RUN-001", "Chief Fact Checker & Quality Assurance Auditor", "Completed", tokens_used=None, duration_s=0.8)
    extract_and_save_citations_from_report("RUN-001", initial_report)
    existing_runs = get_all_runs()

# Custom CSS for polished layout, cards, and badges
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-demo {
        background-color: #ff4b4b20;
        color: #ff4b4b;
        border: 1px solid #ff4b4b50;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-live {
        background-color: #21c35420;
        color: #21c354;
        border: 1px solid #21c35450;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .status-approved {
        color: #21c354;
        font-weight: 700;
    }
    .status-rejected {
        color: #ff4b4b;
        font-weight: 700;
    }
    .status-pending {
        color: #ffa421;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Management
# ---------------------------------------------------------
if "active_run_id" not in st.session_state:
    st.session_state.active_run_id = existing_runs[0]["run_id"] if existing_runs else "RUN-001"

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"


# ---------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("CrewAI • Google Gemini • SQLite")

    pages = [
        "📊 Dashboard",
        "🔬 New Research",
        "📑 Research Results",
        "🕒 Run History",
        "📈 Analytics",
        "ℹ️ About Project"
    ]

    selected_page_raw = st.radio(
        "Navigation",
        pages,
        index=pages.index(f"📊 {st.session_state.nav_page}") if f"📊 {st.session_state.nav_page}" in pages else 0
    )
    current_page = selected_page_raw.split(" ", 1)[1]
    st.session_state.nav_page = current_page

    st.divider()

    # System Environment Status (No API keys exposed)
    st.subheader("⚙️ System Status")
    env_status = check_environment_status()
    gemini_ok = env_status.get("GEMINI_API_KEY", False)
    serper_ok = env_status.get("SERPER_API_KEY", False)

    col1, col2 = st.columns(2)
    with col1:
        if gemini_ok:
            st.success("Gemini: Ready", icon="✅")
        else:
            st.error("Gemini: Missing", icon="❌")
    with col2:
        if serper_ok:
            st.success("Serper: Ready", icon="✅")
        else:
            st.error("Serper: Missing", icon="❌")

    if gemini_ok and serper_ok:
        st.caption("🟢 Live Mode & Demo Mode available")
    else:
        st.caption("🟡 Demo Mode fully functional; Live Mode requires .env credentials.")

    st.caption(f"🤖 LLM Model: `{GEMINI_MODEL_ID}`")

    st.divider()
    st.caption("💾 **Storage**: SQLite Persistent (`research_assistant.db`)")
    st.caption("**Learner:** Swati Mittal | Roll No: 2300320100260")


# ---------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------
if current_page == "Dashboard":
    st.title("📊 Multi-Agent Research Assistant Dashboard")
    st.markdown("""
    Welcome to the **Multi-Agent Research Assistant** built with **CrewAI**, **Google Gemini**, and **persistent SQLite storage**.
    The system deploys three specialized autonomous agents collaborating sequentially:
    **Senior Research Analyst** ➔ **Lead Technical Writer** ➔ **Chief Fact Checker** ➔ **Mandatory Human Review**.
    """)

    # Query metrics live from SQLite database
    metrics = get_run_metrics()
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Runs</div><div class="metric-value">{metrics["total_runs"]}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Completed</div><div class="metric-value" style="color:#21c354;">{metrics["completed_runs"]}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Failed</div><div class="metric-value" style="color:#ff4b4b;">{metrics["failed_runs"]}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Approved</div><div class="metric-value" style="color:#21c354;">{metrics["approved_runs"]}</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Pending Review</div><div class="metric-value" style="color:#ffa421;">{metrics["pending_runs"]}</div></div>', unsafe_allow_html=True)

    st.subheader("⚡ Quick Actions")
    c_btn1, c_btn2, c_space = st.columns([2, 2, 4])
    with c_btn1:
        if st.button("🚀 Start New Research", type="primary", use_container_width=True):
            st.session_state.nav_page = "New Research"
            st.rerun()
    with c_btn2:
        if st.button("📑 View Active Report", use_container_width=True):
            st.session_state.nav_page = "Research Results"
            st.rerun()

    st.subheader("🕒 Persistent Run History (SQLite)")
    db_runs = get_all_runs()
    if db_runs:
        table_rows = []
        for r in db_runs[:10]:
            table_rows.append({
                "Run ID": r["run_id"],
                "Created At": r["created_at"],
                "Topic": r["topic"],
                "Execution Mode": f"🧪 {r['execution_mode'].upper()}" if r["execution_mode"] == "demo" else f"🟢 {r['execution_mode'].upper()}",
                "Status": "✅ Completed" if r["status"] == "Completed" else "❌ Failed",
                "Human Review": r.get("review_status", "Pending Review")
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No research runs recorded yet. Start a new run to view activity.")

    st.subheader("🧩 Architecture & Sequential Pipeline")
    st.markdown("""
    ```mermaid
    flowchart LR
        UserTopic["User Research Topic"] --> Researcher["1. Researcher Agent\n(Serper Google Search)"]
        Researcher -->|"Structured Research Notes & URLs"| Writer["2. Writer Agent\n(8-Section Report Drafting)"]
        Writer -->|"Draft Report + Notes"| FactChecker["3. Fact Checker Agent\n(Audits Claims & Verification Table)"]
        FactChecker -->|"Corrected Report + Audit"| SQLiteDB[("SQLite Storage\n(research_runs,\nagent_executions,\ncitations)")]
        SQLiteDB --> HumanReview["4. Human Reviewer\n(Mandatory Approval / Audit)"]
    ```
    """)


# ---------------------------------------------------------
# PAGE 2: NEW RESEARCH
# ---------------------------------------------------------
elif current_page == "New Research":
    st.title("🔬 Launch Multi-Agent Research")
    st.markdown("Enter a topic below to initiate sequential agent collaboration.")

    # Execution Mode Selector
    mode_selection = st.radio(
        "Select Execution Mode:",
        [
            "🧪 Demo Mode (Recommended - Instant local simulation, zero API costs)",
            "🟢 Live Mode (CrewAI Multi-Agent execution with Google Gemini and Serper)"
        ],
        index=0
    )
    is_demo_mode = "Demo Mode" in mode_selection

    if is_demo_mode:
        st.info(
            "💡 **Demo Mode Active**: Uses deterministic local fixtures to illustrate the 3-agent pipeline. "
            "No API keys required, zero network cost, guaranteed safe execution."
        )
    else:
        if not (gemini_ok and serper_ok):
            st.warning(
                "⚠️ **Live Mode Warning**: One or more required API keys are missing. "
                "Ensure `GEMINI_API_KEY` and `SERPER_API_KEY` are configured in `.env` before running Live Mode."
            )

    # Sample topic shortcuts
    st.caption("Quick Topic Suggestions:")
    s_col1, s_col2, s_col3 = st.columns(3)
    preset_topic = ""
    if s_col1.button("🎓 AI in Higher Education"):
        preset_topic = "Responsible Use of Generative AI in Higher Education"
    if s_col2.button("🏥 Autonomous Diagnostic AI"):
        preset_topic = "Autonomous AI Agents in Healthcare Diagnosis"
    if s_col3.button("🌊 Marine Microplastics"):
        preset_topic = "Microplastics in Marine Ecosystems and Trophic Bioaccumulation"

    topic_input = st.text_input(
        "Research Topic",
        value=preset_topic if preset_topic else "",
        placeholder="e.g. Clinical validation of autonomous AI diagnostic agents"
    )

    start_research_clicked = st.button("🚀 Start Multi-Agent Research", type="primary")

    if start_research_clicked:
        try:
            cleaned_topic = validate_topic(topic_input)
            all_existing = get_all_runs()
            run_id = f"RUN-{len(all_existing) + 1:03d}"
            timestamp_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if is_demo_mode:
                st.subheader(f"⚙️ Executing Multi-Agent Pipeline for: {cleaned_topic}")
                
                # Visual Stage Indicators
                progress_bar = st.progress(0)
                status_box = st.empty()

                # Stage 1: Senior Research Analyst
                status_box.markdown("🔍 **[Stage 1/3] Senior Research Analyst**: Querying search tools, locating primary sources, compiling research notes...")
                progress_bar.progress(33)
                time.sleep(0.8)

                # Stage 2: Lead Technical Writer
                status_box.markdown("✍️ **[Stage 2/3] Lead Technical Writer**: Synthesizing evidence, organizing into 8 required sections, anchoring in-text citations...")
                progress_bar.progress(66)
                time.sleep(0.8)

                # Stage 3: Chief Fact Checker
                status_box.markdown("🛡️ **[Stage 3/3] Chief Fact Checker**: Auditing claims against evidence, classifying assertions ([SUPPORTED], [UNRESOLVED]), producing verification summary...")
                progress_bar.progress(100)
                time.sleep(0.6)

                report_output = get_demo_research_report(cleaned_topic)
                timestamp_completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Persist to SQLite database
                save_research_run(
                    run_id=run_id,
                    topic=cleaned_topic,
                    execution_mode="demo",
                    status="Completed",
                    created_at=timestamp_created,
                    completed_at=timestamp_completed,
                    final_report=report_output
                )

                # Record 3 agent execution stages (tokens_used is NULL since simulated)
                save_agent_execution(run_id, "Senior Research Analyst", "Completed", tokens_used=None, duration_s=0.8)
                save_agent_execution(run_id, "Lead Technical & Research Writer", "Completed", tokens_used=None, duration_s=0.8)
                save_agent_execution(run_id, "Chief Fact Checker & Quality Assurance Auditor", "Completed", tokens_used=None, duration_s=0.6)

                # Extract and persist citations
                extract_and_save_citations_from_report(run_id, report_output)

                st.session_state.active_run_id = run_id
                status_box.success("✅ Multi-agent research completed and saved to SQLite! (Demonstration Output)")
                st.balloons()
                st.info("Navigate to **Research Results** to view the full dossier, audit checklist, and complete human review.")

            else:
                # Live Mode Execution
                validate_environment()
                st.subheader(f"🟢 Executing Live CrewAI Agents: {cleaned_topic}")
                with st.spinner("CrewAI agents collaborating (Researcher -> Writer -> Fact Checker). This typically takes 1-3 minutes..."):
                    try:
                        start_time = time.time()
                        live_result = run_research(cleaned_topic)
                        duration_elapsed = time.time() - start_time
                        report_output = str(live_result)
                        timestamp_completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Persist to SQLite
                        save_research_run(
                            run_id=run_id,
                            topic=cleaned_topic,
                            execution_mode="live",
                            status="Completed",
                            created_at=timestamp_created,
                            completed_at=timestamp_completed,
                            final_report=report_output
                        )
                        # Log agent executions (tokens stored as NULL if not natively reported by LiteLLM)
                        save_agent_execution(run_id, "Senior Research Analyst", "Completed", tokens_used=None, duration_s=round(duration_elapsed * 0.4, 2))
                        save_agent_execution(run_id, "Lead Technical & Research Writer", "Completed", tokens_used=None, duration_s=round(duration_elapsed * 0.35, 2))
                        save_agent_execution(run_id, "Chief Fact Checker & Quality Assurance Auditor", "Completed", tokens_used=None, duration_s=round(duration_elapsed * 0.25, 2))

                        extract_and_save_citations_from_report(run_id, report_output)

                        st.session_state.active_run_id = run_id
                        st.success("✅ Live CrewAI research finished and saved to SQLite database!")

                    except Exception as exc:
                        # Record failed run in SQLite
                        save_research_run(
                            run_id=run_id,
                            topic=cleaned_topic,
                            execution_mode="live",
                            status="Failed",
                            created_at=timestamp_created,
                            completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            final_report=f"Execution failed: {exc}"
                        )
                        save_agent_execution(
                            run_id=run_id,
                            agent_role="Workflow Orchestrator",
                            status="Failed",
                            tokens_used=None,
                            duration_s=None,
                            error_message=str(exc)
                        )
                        st.error(f"Live Execution Error: {exc}")
                        st.info("Check your Google AI Studio quota / billing or switch to Demo Mode for instant testing.")

        except TopicValidationError as te:
            st.error(f"Input Error: {te}")
        except ResearchWorkflowError as rwe:
            st.error(f"Workflow Error: {rwe}")
        except EnvironmentError as ee:
            st.error(f"Configuration Error: {ee}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")


# ---------------------------------------------------------
# PAGE 3: RESEARCH RESULTS & HUMAN REVIEW
# ---------------------------------------------------------
elif current_page == "Research Results":
    st.title("📑 Multi-Agent Research Results")

    # Fetch active run from database
    active_run = get_run_by_id(st.session_state.active_run_id)

    if not active_run:
        all_runs = get_all_runs()
        if all_runs:
            active_run = get_run_by_id(all_runs[0]["run_id"])
            st.session_state.active_run_id = all_runs[0]["run_id"]

    if not active_run:
        st.info("No active research report available in SQLite. Run a new research query or select a run from Run History.")
    else:
        # Header Badge & Metadata
        col_meta1, col_meta2 = st.columns([3, 1])
        with col_meta1:
            st.subheader(active_run["topic"])
            st.caption(f"Run ID: **{active_run['run_id']}** | Created: **{active_run['created_at']}** | Completed: **{active_run.get('completed_at', 'N/A')}**")
        with col_meta2:
            if active_run["execution_mode"] == "demo":
                st.markdown('<span class="badge-demo">🧪 DEMO MODE (SYNTHETIC)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-live">🟢 LIVE EXECUTION</span>', unsafe_allow_html=True)

        # Clear Banner for Demo Mode
        if active_run["execution_mode"] == "demo":
            st.warning(
                "⚠️ **DEMONSTRATION OUTPUT (SYNTHETIC / SIMULATED)**\n\n"
                "This report was produced using deterministic demonstration fixtures without making external API calls. "
                "Simulated citations and empirical claims illustrate the multi-agent pipeline architecture only and must not be treated as verified primary research."
            )

        # Report Content Tabs
        tab_full, tab_audit, tab_citations, tab_download = st.tabs([
            "📄 Full Research Dossier",
            "🛡️ Fact-Checking Audit",
            "🔗 Saved Citations (DB)",
            "📥 Export & Downloads"
        ])

        with tab_full:
            st.markdown(active_run["final_report"])

        with tab_audit:
            st.subheader("Auditor Claims Classification")
            report_text = active_run["final_report"] or ""
            if "PART I" in report_text and "PART II" in report_text:
                audit_part = report_text.split("PART II")[0]
                st.markdown(audit_part)
            else:
                st.markdown(report_text)

        with tab_citations:
            st.subheader("Database Citations Log")
            citations = active_run.get("citations", [])
            if citations:
                cit_rows = []
                for c in citations:
                    cit_rows.append({
                        "ID": c["citation_id"],
                        "Source Title": c["source_title"],
                        "URL": c["source_url"],
                        "Status": c["verification_status"]
                    })
                st.dataframe(pd.DataFrame(cit_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No structured citations recorded in database for this run.")

        with tab_download:
            st.subheader("Export Verified Research Report")
            is_demo = active_run["execution_mode"] == "demo"
            current_rev_status = active_run.get("review_status", "Pending Review")

            col_down1, col_down2 = st.columns(2)
            with col_down1:
                # Markdown download
                md_download_text = active_run["final_report"] or ""
                if is_demo:
                    md_download_text = f"> **[DEMONSTRATION ONLY - NOT REAL RESEARCH]**\n\n{md_download_text}"
                st.download_button(
                    "📥 Download Markdown (.md)",
                    data=md_download_text,
                    file_name=f"{active_run['run_id']}_research_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with col_down2:
                # PDF download with watermark
                pdf_bytes = generate_research_pdf(
                    topic=active_run["topic"],
                    markdown_content=active_run["final_report"] or "",
                    is_demo=is_demo,
                    review_status=current_rev_status
                )
                st.download_button(
                    "📄 Download PDF Document (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{active_run['run_id']}_research_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        st.divider()

        # ---------------------------------------------------------
        # Mandatory Human Review Section (Persisted to SQLite)
        # ---------------------------------------------------------
        st.subheader("🧑‍⚖️ Mandatory Human Review Stage")
        st.caption("As required by ethical AI guidelines, an AI-generated report must undergo human review before publication.")

        rev_status = active_run.get("review_status", "Pending Review")
        if rev_status == "Approved":
            st.markdown(f'Current Status: <span class="status-approved">✅ APPROVED BY HUMAN REVIEWER</span> ({active_run.get("review_timestamp", "")})', unsafe_allow_html=True)
        elif rev_status == "Rejected":
            st.markdown(f'Current Status: <span class="status-rejected">❌ REJECTED BY HUMAN REVIEWER</span> ({active_run.get("review_timestamp", "")})', unsafe_allow_html=True)
        else:
            st.markdown('Current Status: <span class="status-pending">⏳ PENDING HUMAN AUDIT</span>', unsafe_allow_html=True)

        user_comments = st.text_area(
            "Reviewer Comments & Feedback",
            value=active_run.get("reviewer_comments", ""),
            placeholder="Document any factual corrections, citation concerns, or notes for approval..."
        )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 4])
        with btn_col1:
            if st.button("✅ Approve Report", type="primary", use_container_width=True):
                save_human_review(
                    run_id=active_run["run_id"],
                    decision="Approved",
                    comments=user_comments
                )
                st.success("Report approved and persisted to SQLite!")
                st.rerun()

        with btn_col2:
            if st.button("❌ Reject Report", use_container_width=True):
                save_human_review(
                    run_id=active_run["run_id"],
                    decision="Rejected",
                    comments=user_comments
                )
                st.warning("Report rejected and persisted to SQLite.")
                st.rerun()


# ---------------------------------------------------------
# PAGE 4: RUN HISTORY (SQLite)
# ---------------------------------------------------------
elif current_page == "Run History":
    st.title("🕒 Research Execution History (SQLite)")
    st.markdown("All research runs, agent execution logs, and human review audits are permanently stored in SQLite.")

    db_runs = get_all_runs()
    if not db_runs:
        st.info("No runs recorded in database yet.")
    else:
        for idx, r in enumerate(db_runs):
            with st.container():
                c1, c2, c3, c4 = st.columns([1.5, 3.5, 2, 2])
                with c1:
                    st.write(f"**{r['run_id']}**")
                    st.caption(r["created_at"])
                with c2:
                    st.write(f"**{r['topic']}**")
                    if r["execution_mode"] == "demo":
                        st.markdown('<span class="badge-demo">Demo Mode</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-live">Live Mode</span>', unsafe_allow_html=True)
                with c3:
                    rev = r.get("review_status", "Pending Review")
                    if rev == "Approved":
                        st.markdown('<span class="status-approved">✅ Approved</span>', unsafe_allow_html=True)
                    elif rev == "Rejected":
                        st.markdown('<span class="status-rejected">❌ Rejected</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-pending">⏳ Pending Review</span>', unsafe_allow_html=True)
                with c4:
                    if st.button("Open Dossier", key=f"btn_open_{r['run_id']}"):
                        st.session_state.active_run_id = r["run_id"]
                        st.session_state.nav_page = "Research Results"
                        st.rerun()
                st.divider()


# ---------------------------------------------------------
# PAGE 5: ANALYTICS & BENCHMARKS
# ---------------------------------------------------------
elif current_page == "Analytics":
    st.title("📈 Multi-Agent Analytics & Benchmark Hub")
    st.markdown("Explore empirical performance benchmarks, agent role comparisons, and live SQLite database statistics.")

    # Two clearly partitioned tabs to strictly prevent mixing synthetic and live metrics
    tab_synthetic, tab_live_db = st.tabs([
        "📊 Synthetic Benchmark Analytics (dataset.csv)",
        "💾 Production Application Metrics (SQLite DB)"
    ])

    # ---------------------------------------------------------
    # SUB-TAB 1: SYNTHETIC BENCHMARK DATASET ANALYTICS
    # ---------------------------------------------------------
    with tab_synthetic:
        st.warning(
            "⚠️ **SYNTHETIC BENCHMARK DATASET NOTICE**\n\n"
            "`dataset.csv` is an entirely synthetic 5,000-row demonstration dataset generated with random seed 78 for "
            "workflow evaluation and multi-agent performance simulation. It must NOT be represented as real production execution logs."
        )

        try:
            df = load_dataset()
            kpis = compute_overview_kpis(df)

            # 1. Dataset Overview KPI Metrics (7 required indicators)
            st.subheader("📌 Benchmark Dataset Overview (5,000 Records)")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-value">{kpis["total_records"]:,}</div></div>', unsafe_allow_html=True)
            with k2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Research Topics</div><div class="metric-value" style="color:#3b82f6;">{kpis["total_topics"]}</div></div>', unsafe_allow_html=True)
            with k3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Agent Roles</div><div class="metric-value" style="color:#8b5cf6;">{len(kpis["agent_roles"])}</div></div>', unsafe_allow_html=True)
            with k4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Simulated Fault Rate</div><div class="metric-value" style="color:#ef4444;">{kpis["error_rate_pct"]:.2f}%</div></div>', unsafe_allow_html=True)

            k5, k6, k7, k8 = st.columns(4)
            with k5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Average Tokens Used</div><div class="metric-value">{kpis["avg_tokens"]:,.0f}</div></div>', unsafe_allow_html=True)
            with k6:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Average Duration</div><div class="metric-value">{kpis["avg_duration_s"]:.1f}s</div></div>', unsafe_allow_html=True)
            with k7:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Quality Score</div><div class="metric-value" style="color:#10b981;">{kpis["avg_quality_score"]:.3f}</div></div>', unsafe_allow_html=True)
            with k8:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Simulated Errors</div><div class="metric-value" style="color:#f59e0b;">{kpis["total_errors"]}</div></div>', unsafe_allow_html=True)

            st.divider()

            # 3. Interactive Filter Controls
            st.subheader("🔍 Interactive Multi-Criteria Filters")
            with st.expander("Filter Controls (Agent Role, Research Topic, Error Status)", expanded=True):
                f_col1, f_col2, f_col3 = st.columns([2, 3, 2])

                all_roles = df["agent_role"].unique().tolist()
                with f_col1:
                    selected_roles = st.multiselect(
                        "Agent Role(s)",
                        options=all_roles,
                        default=all_roles
                    )

                all_topics = sorted(df["task_type"].unique().tolist())
                with f_col2:
                    selected_topics = st.multiselect(
                        "Research Topic(s)",
                        options=all_topics,
                        default=all_topics[:5],
                        help="Select one or more of the 20 benchmark research topics."
                    )

                with f_col3:
                    error_filter = st.radio(
                        "Error Status",
                        options=["All", "Error-Free Only (0)", "Errors Only (1)"],
                        index=0
                    )

            # Apply filters
            filtered_df = filter_dataset(
                df,
                roles=selected_roles if selected_roles else all_roles,
                topics=selected_topics if selected_topics else all_topics,
                error_status=error_filter
            )

            filter_pct = (len(filtered_df) / len(df)) * 100
            st.caption(f"**Filtered Subset**: Displaying **{len(filtered_df):,}** of **{len(df):,}** records ({filter_pct:.1f}%).")

            # 2. Visualizations (All 5 required charts)
            st.subheader("📊 Interactive Performance Visualizations")

            # Chart Row 1: Tokens by Agent and Duration by Agent
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                fig_tokens = plot_token_usage_by_agent(filtered_df)
                st.plotly_chart(fig_tokens, use_container_width=True)
            with v_col2:
                fig_dur = plot_duration_by_agent(filtered_df)
                st.plotly_chart(fig_dur, use_container_width=True)

            # Chart Row 2: Error Rate by Agent and Token Distribution
            v_col3, v_col4 = st.columns(2)
            with v_col3:
                fig_err = plot_error_rate_by_agent(filtered_df)
                st.plotly_chart(fig_err, use_container_width=True)
            with v_col4:
                fig_hist = plot_token_distribution(filtered_df)
                st.plotly_chart(fig_hist, use_container_width=True)

            # Chart Row 3: Quality Score by Topic
            st.plotly_chart(plot_quality_by_topic(filtered_df), use_container_width=True)

            st.divider()

            # 4. Searchable Dataset Preview & Filtered CSV Download
            st.subheader("📋 Searchable Benchmark Dataset Preview")
            c_search, c_download = st.columns([3, 1])

            with c_search:
                search_query = st.text_input("Search records by Run ID, Topic, or Role", placeholder="e.g. Researcher, HealthAI, R00100")

            display_preview = filtered_df
            if search_query.strip():
                q = search_query.strip().lower()
                display_preview = filtered_df[
                    filtered_df["run_id"].str.lower().str.contains(q) |
                    filtered_df["task_type"].str.lower().str.contains(q) |
                    filtered_df["agent_role"].str.lower().str.contains(q)
                ]

            with c_download:
                st.write("")
                st.write("")
                csv_bytes = display_preview.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Filtered CSV",
                    data=csv_bytes,
                    file_name=f"filtered_synthetic_dataset_{len(display_preview)}_rows.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.dataframe(display_preview.head(250), use_container_width=True)
            st.caption(f"Showing up to 250 records matching active filters and search term. (Total matching: {len(display_preview):,})")

            # Data Dictionary Reference
            with st.expander("📖 View Data Dictionary (`data_dictionary.csv`)"):
                try:
                    data_dict = load_data_dictionary()
                    st.dataframe(data_dict, use_container_width=True, hide_index=True)
                except Exception as ex_dict:
                    st.warning(f"Data dictionary not available: {ex_dict}")

        except Exception as ex_load:
            st.error(f"Failed to load dataset: {ex_load}")

    # ---------------------------------------------------------
    # SUB-TAB 2: REAL APPLICATION EXECUTION METRICS (SQLITE DB)
    # ---------------------------------------------------------
    with tab_live_db:
        st.subheader("💾 Actual Application Execution Metrics (SQLite Storage)")
        st.markdown(
            "These metrics reflect **real application activity** stored in the local SQLite database (`research_assistant.db`). "
            "They represent user-initiated Demo and Live runs and are maintained completely separate from synthetic benchmark records."
        )

        db_metrics = get_run_metrics()
        conn = get_connection()

        # Summary KPIs for SQLite application runs
        db_k1, db_k2, db_k3, db_k4 = st.columns(4)
        with db_k1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total SQLite Runs</div><div class="metric-value">{db_metrics["total_runs"]}</div></div>', unsafe_allow_html=True)
        with db_k2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Completed Runs</div><div class="metric-value" style="color:#21c354;">{db_metrics["completed_runs"]}</div></div>', unsafe_allow_html=True)
        with db_k3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Failed Runs</div><div class="metric-value" style="color:#ff4b4b;">{db_metrics["failed_runs"]}</div></div>', unsafe_allow_html=True)
        with db_k4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Approved Runs</div><div class="metric-value" style="color:#21c354;">{db_metrics["approved_runs"]}</div></div>', unsafe_allow_html=True)

        # Mode distribution in SQLite
        cur_modes = conn.execute("SELECT execution_mode, COUNT(*) as count FROM research_runs GROUP BY execution_mode;")
        mode_counts = {row["execution_mode"]: row["count"] for row in cur_modes.fetchall()}

        # Total citations & agent executions logged
        cur_cit = conn.execute("SELECT COUNT(*) FROM citations;")
        total_citations = cur_cit.fetchone()[0]

        cur_exec = conn.execute("SELECT COUNT(*) FROM agent_executions;")
        total_execs = cur_exec.fetchone()[0]

        db_k5, db_k6, db_k7 = st.columns(3)
        with db_k5:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Demo Runs vs Live Runs</div><div class="metric-value">{mode_counts.get("demo", 0)} / {mode_counts.get("live", 0)}</div></div>', unsafe_allow_html=True)
        with db_k6:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Logged Executions</div><div class="metric-value">{total_execs}</div></div>', unsafe_allow_html=True)
        with db_k7:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Citations Tracked</div><div class="metric-value" style="color:#3b82f6;">{total_citations}</div></div>', unsafe_allow_html=True)

        st.divider()

        # Database Tables Overview
        st.subheader("📑 SQLite Tables Live Inspection")
        tab_db_runs, tab_db_execs, tab_db_cit, tab_db_rev = st.tabs([
            "1. research_runs",
            "2. agent_executions",
            "3. citations",
            "4. human_reviews"
        ])

        with tab_db_runs:
            df_runs = pd.read_sql_query("SELECT run_id, topic, execution_mode, status, created_at, completed_at FROM research_runs ORDER BY created_at DESC;", conn)
            st.dataframe(df_runs, use_container_width=True)

        with tab_db_execs:
            df_execs = pd.read_sql_query("SELECT execution_id, run_id, agent_role, tokens_used, duration_s, status, error_message FROM agent_executions ORDER BY execution_id DESC;", conn)
            st.dataframe(df_execs, use_container_width=True)
            st.caption("Note: `tokens_used` is stored as `NULL` when unmeasured by the provider, fulfilling academic transparency.")

        with tab_db_cit:
            df_cit = pd.read_sql_query("SELECT citation_id, run_id, source_title, source_url, verification_status FROM citations ORDER BY citation_id DESC;", conn)
            st.dataframe(df_cit, use_container_width=True)

        with tab_db_rev:
            df_rev = pd.read_sql_query("SELECT review_id, run_id, decision, comments, reviewed_at FROM human_reviews ORDER BY review_id DESC;", conn)
            st.dataframe(df_rev, use_container_width=True)

        conn.close()


# ---------------------------------------------------------
# PAGE 6: ABOUT PROJECT
# ---------------------------------------------------------
elif current_page == "About Project":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Project 78: Multi-Agent Research Assistant with CrewAI
    - **Learner:** Swati Mittal
    - **Roll Number:** 2300320100260
    - **Campus:** ABES Engineering College, Ghaziabad
    - **Programme:** HCLTech Career Shaper Programme | 2025–26

    ---

    ### Workflow Architecture
    The application utilizes a sequential CrewAI multi-agent team:
    1. **Senior Research Analyst**: Conducts targeted web research using Google Serper Dev Tool, records canonical source URLs, publication dates, and key empirical findings.
    2. **Lead Technical Writer**: Receives research notes via task context and structures an 8-section report with in-text markdown citations.
    3. **Chief Fact Checker**: Audits claims against source evidence, classifies statements as `[SUPPORTED]`, `[UNSUPPORTED]`, or `[UNRESOLVED]`, compiles a Verification Summary Table, and corrects the final report.
    4. **Human Reviewer**: Mandatory human-in-the-loop review ensuring accuracy and ethical accountability.

    ---

    ### Database Schema (SQLite: `research_assistant.db`)
    - `research_runs`: Stores run metadata, execution mode (`demo`/`live`), status, and final report text.
    - `agent_executions`: Records per-agent execution logs, duration, status, and tokens (NULL when unmeasured).
    - `citations`: Tracks extracted citations, source titles, canonical URLs, and fact-checking verification status.
    - `human_reviews`: Stores human review audit decisions (`Approved`/`Rejected`), reviewer feedback, and timestamps.

    ---

    ### Ethical Compliance & Academic Integrity
    - **Zero Invented Citations**: Agents are constrained by strict system prompts to never fabricate URLs, authors, or statistics.
    - **Demonstration Disclosures**: Demo mode outputs are clearly labeled as synthetic artifacts to prevent misrepresentation as verified real research.
    - **Secret Protection**: API keys are loaded via `python-dotenv` and are never exposed in logs, database tables, UI elements, or generated PDFs.
    """)
