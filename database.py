"""
Database Persistence Module for Multi-Agent Research Assistant.
SQLite schema management, safe parameterized queries, and persistence services.
"""
import os
import re
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

# Default database file in the project workspace
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_assistant.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Establish a connection to the SQLite database with row factory and foreign keys enabled.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Automatically creates the required tables and indexes on first launch.
    Uses safe, idempotent DDL (CREATE TABLE IF NOT EXISTS).
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. research_runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS research_runs (
        run_id TEXT PRIMARY KEY,
        topic TEXT NOT NULL,
        execution_mode TEXT NOT NULL CHECK(execution_mode IN ('demo', 'live')),
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        final_report TEXT
    );
    """)

    # 2. agent_executions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_executions (
        execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        agent_role TEXT NOT NULL,
        tokens_used INTEGER,
        duration_s REAL,
        status TEXT NOT NULL,
        error_message TEXT,
        FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
    );
    """)

    # 3. citations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS citations (
        citation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        source_title TEXT NOT NULL,
        source_url TEXT NOT NULL,
        verification_status TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
    );
    """)

    # 4. human_reviews table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS human_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        decision TEXT NOT NULL CHECK(decision IN ('Approved', 'Rejected')),
        comments TEXT,
        reviewed_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
    );
    """)

    # Indexes for performance on foreign keys and timestamps
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_exec_run_id ON agent_executions(run_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_run_id ON citations(run_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_human_reviews_run_id ON human_reviews(run_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_research_runs_created ON research_runs(created_at);")

    conn.commit()
    conn.close()


def save_research_run(
    run_id: str,
    topic: str,
    execution_mode: str,
    status: str,
    created_at: str,
    completed_at: Optional[str] = None,
    final_report: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Inserts or updates a research run using parameterized SQL.
    Ensures execution_mode is stored in lowercase ('demo' or 'live').
    """
    mode = execution_mode.lower().strip()
    if "demo" in mode:
        normalized_mode = "demo"
    else:
        normalized_mode = "live"

    conn = get_connection(db_path)
    with conn:
        conn.execute("""
        INSERT INTO research_runs (run_id, topic, execution_mode, status, created_at, completed_at, final_report)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            topic = excluded.topic,
            execution_mode = excluded.execution_mode,
            status = excluded.status,
            completed_at = excluded.completed_at,
            final_report = excluded.final_report;
        """, (run_id, topic, normalized_mode, status, created_at, completed_at, final_report))
    conn.close()


def save_agent_execution(
    run_id: str,
    agent_role: str,
    status: str,
    tokens_used: Optional[int] = None,
    duration_s: Optional[float] = None,
    error_message: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Inserts an agent execution log record.
    CRITICAL: Unmeasured token counts MUST be stored as NULL, never fabricated.
    """
    # Ensure tokens_used is strictly integer or None (NULL in SQL)
    if tokens_used is not None:
        try:
            tokens_used = int(tokens_used)
        except (ValueError, TypeError):
            tokens_used = None

    if duration_s is not None:
        try:
            duration_s = float(duration_s)
        except (ValueError, TypeError):
            duration_s = None

    conn = get_connection(db_path)
    with conn:
        cur = conn.execute("""
        INSERT INTO agent_executions (run_id, agent_role, tokens_used, duration_s, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?);
        """, (run_id, agent_role, tokens_used, duration_s, status, error_message))
        execution_id = cur.lastrowid
    conn.close()
    return execution_id


def save_citation(
    run_id: str,
    source_title: str,
    source_url: str,
    verification_status: str,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Inserts a single citation record linked to a run_id using parameterized query.
    """
    conn = get_connection(db_path)
    with conn:
        cur = conn.execute("""
        INSERT INTO citations (run_id, source_title, source_url, verification_status)
        VALUES (?, ?, ?, ?);
        """, (run_id, source_title.strip(), source_url.strip(), verification_status.strip()))
        citation_id = cur.lastrowid
    conn.close()
    return citation_id


def extract_and_save_citations_from_report(run_id: str, report_text: str, db_path: str = DEFAULT_DB_PATH) -> List[int]:
    """
    Parses verification summary and references in the report text to persist citations.
    """
    if not report_text:
        return []

    citation_ids = []
    # Pattern 1: Parse claims table rows containing [Source Title](URL) and status
    table_pattern = re.compile(r"\|\s*([^|]+)\s*\|\s*\[([^\]]+)\]\((https?://[^)]+)\)\s*\|\s*(\[[A-Z]+\]|[A-Z]+)\s*\|", re.MULTILINE)
    matches = table_pattern.findall(report_text)

    for match in matches:
        _, title, url, status = match
        cid = save_citation(run_id, title, url, status, db_path=db_path)
        citation_ids.append(cid)

    # Pattern 2: If no table matched, parse standard references: 1. Title. URL
    if not citation_ids:
        ref_pattern = re.compile(r"^\d+\.\s*(.*?)(https?://[^\s)]+)", re.MULTILINE)
        for ref_match in ref_pattern.findall(report_text):
            title, url = ref_match
            title_clean = title.strip().rstrip(".:- ") or "Referenced Web Source"
            cid = save_citation(run_id, title_clean, url.strip(), "[SUPPORTED]", db_path=db_path)
            citation_ids.append(cid)

    return citation_ids


def save_human_review(
    run_id: str,
    decision: str,
    comments: Optional[str] = None,
    reviewed_at: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Inserts a human review audit decision ('Approved' or 'Rejected').
    """
    if decision not in ("Approved", "Rejected"):
        raise ValueError(f"Invalid review decision '{decision}'. Must be 'Approved' or 'Rejected'.")

    timestamp_str = reviewed_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection(db_path)
    with conn:
        cur = conn.execute("""
        INSERT INTO human_reviews (run_id, decision, comments, reviewed_at)
        VALUES (?, ?, ?, ?);
        """, (run_id, decision, comments or "", timestamp_str))
        review_id = cur.lastrowid
    conn.close()
    return review_id


def get_all_runs(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Returns all runs ordered by created_at DESC, joined with their latest human review decision.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    query = """
    SELECT 
        r.run_id,
        r.topic,
        r.execution_mode,
        r.status,
        r.created_at,
        r.completed_at,
        r.final_report,
        hr.decision AS review_status,
        hr.comments AS reviewer_comments,
        hr.reviewed_at AS review_timestamp
    FROM research_runs r
    LEFT JOIN (
        SELECT run_id, decision, comments, reviewed_at,
               ROW_NUMBER() OVER(PARTITION BY run_id ORDER BY review_id DESC) as rn
        FROM human_reviews
    ) hr ON r.run_id = hr.run_id AND hr.rn = 1
    ORDER BY r.created_at DESC;
    """
    cursor = conn.execute(query)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Normalize defaults
    for r in rows:
        if not r["review_status"]:
            r["review_status"] = "Pending Review"
    return rows


def get_run_by_id(run_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves full details for a single run, including its agent executions, citations, and reviews.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.execute("SELECT * FROM research_runs WHERE run_id = ?;", (run_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    run_dict = dict(row)

    # Fetch executions
    cur_exec = conn.execute(
        "SELECT * FROM agent_executions WHERE run_id = ? ORDER BY execution_id ASC;", (run_id,)
    )
    run_dict["agent_executions"] = [dict(e) for e in cur_exec.fetchall()]

    # Fetch citations
    cur_cit = conn.execute(
        "SELECT * FROM citations WHERE run_id = ? ORDER BY citation_id ASC;", (run_id,)
    )
    run_dict["citations"] = [dict(c) for c in cur_cit.fetchall()]

    # Fetch human reviews
    cur_rev = conn.execute(
        "SELECT * FROM human_reviews WHERE run_id = ? ORDER BY review_id DESC;", (run_id,)
    )
    reviews = [dict(rv) for rv in cur_rev.fetchall()]
    run_dict["reviews"] = reviews

    if reviews:
        run_dict["review_status"] = reviews[0]["decision"]
        run_dict["reviewer_comments"] = reviews[0]["comments"]
        run_dict["review_timestamp"] = reviews[0]["reviewed_at"]
    else:
        run_dict["review_status"] = "Pending Review"
        run_dict["reviewer_comments"] = ""
        run_dict["review_timestamp"] = None

    conn.close()
    return run_dict


def get_run_metrics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """
    Computes summary metrics directly from SQLite database.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM research_runs;")
    total_runs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM research_runs WHERE status = 'Completed';")
    completed_runs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM research_runs WHERE status = 'Failed';")
    failed_runs = cur.fetchone()[0]

    cur.execute("""
    SELECT COUNT(DISTINCT r.run_id) 
    FROM research_runs r
    JOIN (
        SELECT run_id, decision, ROW_NUMBER() OVER(PARTITION BY run_id ORDER BY review_id DESC) as rn 
        FROM human_reviews
    ) hr ON r.run_id = hr.run_id AND hr.rn = 1
    WHERE hr.decision = 'Approved';
    """)
    approved_runs = cur.fetchone()[0]

    cur.execute("""
    SELECT COUNT(DISTINCT r.run_id) 
    FROM research_runs r
    JOIN (
        SELECT run_id, decision, ROW_NUMBER() OVER(PARTITION BY run_id ORDER BY review_id DESC) as rn 
        FROM human_reviews
    ) hr ON r.run_id = hr.run_id AND hr.rn = 1
    WHERE hr.decision = 'Rejected';
    """)
    rejected_runs = cur.fetchone()[0]

    pending_runs = total_runs - (approved_runs + rejected_runs)

    conn.close()
    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "approved_runs": approved_runs,
        "rejected_runs": rejected_runs,
        "pending_runs": max(0, pending_runs)
    }
