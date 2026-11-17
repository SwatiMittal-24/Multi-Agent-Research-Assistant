"""
PDF Report Generator for Multi-Agent Research Assistant.
Converts research reports into formatted PDFs with human review status and demo watermarking.
"""
import io
import re
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class ResearchReportPDF(FPDF):
    """Custom FPDF class with headers, footers, and demonstration watermarks."""

    def __init__(self, topic: str, is_demo: bool = True, review_status: str = "Pending Review"):
        super().__init__()
        self.topic = topic
        self.is_demo = is_demo
        self.review_status = review_status
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 100, 100)
        self.cell(100, 8, "Multi-Agent Research Assistant (CrewAI)", border=False, new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        if self.is_demo:
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(220, 50, 50)
            self.cell(0, 8, "[DEMO MODE - SYNTHETIC DATA]", border=False, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(40, 140, 40)
            self.cell(0, 8, "[LIVE VERIFIED RUN]", border=False, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_draw_color(200, 200, 200)
        self.line(10, 16, 200, 16)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        generated_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10, f"Page {self.page_no()} | Generated: {generated_date} | Review Status: {self.review_status}", align="C")


def clean_markdown_text(text: str) -> str:
    """Strip or simplify Markdown symbols for clean PDF rendering."""
    # Replace markdown links [Title](URL) with Title (URL)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Remove bold/italic asterisks
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    # Normalize bullet points and dashes
    cleaned = cleaned.replace("•", "-").replace("—", "--").replace("–", "-")
    # Replace unicode quotes with ASCII quotes
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return cleaned


def generate_research_pdf(topic: str, markdown_content: str, is_demo: bool = True, review_status: str = "Pending Review") -> bytes:
    """
    Builds a styled PDF document from report text.
    Returns PDF bytes suitable for st.download_button.
    """
    pdf = ResearchReportPDF(topic=topic, is_demo=is_demo, review_status=review_status)
    pdf.add_page()

    # Title & Metadata Card
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 8, f"Research Report: {topic}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Review Status Badge
    pdf.set_font("Helvetica", "B", 10)
    if "Approved" in review_status:
        pdf.set_text_color(30, 130, 60)
        pdf.cell(0, 6, "Human Review: APPROVED", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    elif "Rejected" in review_status:
        pdf.set_text_color(200, 40, 40)
        pdf.cell(0, 6, "Human Review: REJECTED", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.set_text_color(180, 110, 20)
        pdf.cell(0, 6, "Human Review: PENDING AUDIT", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Demo Banner if applicable
    if is_demo:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(200, 30, 30)
        pdf.multi_cell(
            0, 5,
            "WARNING: This document was produced in DEMONSTRATION MODE without live API calls. "
            "Citations and evidence are synthetic and for workflow evaluation only.",
            border=1,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT
        )
        pdf.ln(3)

    pdf.ln(4)

    # Parse and write content line by line
    cleaned_content = clean_markdown_text(markdown_content)
    lines = cleaned_content.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(3)
            continue

        # Major Headings
        if stripped.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(20, 40, 80)
            pdf.multi_cell(0, 7, stripped[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(30, 60, 110)
            pdf.multi_cell(0, 6, stripped[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
        elif stripped.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 5, stripped[4:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Table rows or bullet points
        elif stripped.startswith("|"):
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(50, 50, 50)
            # Truncate long table rows to fit margins
            pdf.cell(0, 4, stripped[:105], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 5, f"  * {stripped[2:]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 5, stripped, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
