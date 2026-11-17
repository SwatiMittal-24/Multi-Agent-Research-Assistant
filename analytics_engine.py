"""
Analytics Engine for Multi-Agent Research Assistant.
Handles computation, interactive filtering, and visualization generation for dataset.csv
and SQLite application execution metrics.
"""
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional

# Default dataset path
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
DEFAULT_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_dictionary.csv")

# Professional consistent color palette matching Streamlit dark/light UI
COLOR_PRIMARY = "#3b82f6"       # Blue
COLOR_ACCENT = "#10b981"        # Emerald green
COLOR_WARNING = "#f59e0b"       # Amber
COLOR_DANGER = "#ef4444"        # Red
COLOR_PURPLE = "#8b5cf6"        # Violet
AGENT_COLORS = {
    "Researcher": "#3b82f6",
    "Writer": "#8b5cf6",
    "Fact_Checker": "#10b981"
}


def load_dataset(csv_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Loads dataset.csv in read-only mode without mutating the original file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    return pd.read_csv(csv_path)


def load_data_dictionary(dict_path: str = DEFAULT_DICT_PATH) -> pd.DataFrame:
    """
    Loads data_dictionary.csv.
    """
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Data dictionary not found at: {dict_path}")
    return pd.read_csv(dict_path)


def compute_overview_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level statistical metrics from the 5,000-record dataset.
    """
    total_records = len(df)
    if total_records == 0:
        return {
            "total_records": 0,
            "total_topics": 0,
            "agent_roles": [],
            "avg_tokens": 0.0,
            "avg_duration_s": 0.0,
            "avg_quality_score": 0.0,
            "total_errors": 0,
            "error_rate_pct": 0.0
        }

    total_topics = int(df["task_type"].nunique())
    agent_roles = df["agent_role"].unique().tolist()
    avg_tokens = float(round(df["tokens_used"].mean(), 2))
    avg_duration_s = float(round(df["duration_s"].mean(), 2))
    avg_quality_score = float(round(df["quality_score"].mean(), 3))
    total_errors = int(df["errors"].sum())
    error_rate_pct = float(round((total_errors / total_records) * 100, 2))

    return {
        "total_records": total_records,
        "total_topics": total_topics,
        "agent_roles": agent_roles,
        "avg_tokens": avg_tokens,
        "avg_duration_s": avg_duration_s,
        "avg_quality_score": avg_quality_score,
        "total_errors": total_errors,
        "error_rate_pct": error_rate_pct
    }


def filter_dataset(
    df: pd.DataFrame,
    roles: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    error_status: str = "All"
) -> pd.DataFrame:
    """
    Applies multi-criteria filtering to the dataset.
    Returns a filtered copy without modifying the original.
    """
    filtered = df.copy()

    if roles:
        filtered = filtered[filtered["agent_role"].isin(roles)]

    if topics:
        filtered = filtered[filtered["task_type"].isin(topics)]

    if error_status == "Errors Only (1)":
        filtered = filtered[filtered["errors"] == 1]
    elif error_status == "Error-Free Only (0)":
        filtered = filtered[filtered["errors"] == 0]

    return filtered


def plot_token_usage_by_agent(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive bar chart of average token usage per agent role.
    """
    if df.empty:
        return go.Figure()

    grouped = df.groupby("agent_role")["tokens_used"].mean().reset_index()
    grouped.columns = ["agent_role", "avg_tokens"]

    fig = go.Figure(data=[
        go.Bar(
            x=grouped["agent_role"],
            y=grouped["avg_tokens"],
            text=[f"{v:,.0f}" for v in grouped["avg_tokens"]],
            textposition="auto",
            marker=dict(
                color=[AGENT_COLORS.get(role, COLOR_PRIMARY) for role in grouped["agent_role"]],
                line=dict(width=1, color="rgba(0,0,0,0.2)")
            ),
            hovertemplate="<b>Agent Role:</b> %{x}<br><b>Mean Tokens:</b> %{y:,.1f}<extra></extra>"
        )
    ])
    fig.update_layout(
        title="Average Token Consumption by Agent Role",
        xaxis_title="Agent Role",
        yaxis_title="Average Tokens Used",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_duration_by_agent(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive bar chart of average execution duration (s) per agent role.
    """
    if df.empty:
        return go.Figure()

    grouped = df.groupby("agent_role")["duration_s"].mean().reset_index()
    grouped.columns = ["agent_role", "avg_duration"]

    fig = go.Figure(data=[
        go.Bar(
            x=grouped["agent_role"],
            y=grouped["avg_duration"],
            text=[f"{v:.1f}s" for v in grouped["avg_duration"]],
            textposition="auto",
            marker=dict(
                color=[AGENT_COLORS.get(role, COLOR_PURPLE) for role in grouped["agent_role"]],
                line=dict(width=1, color="rgba(0,0,0,0.2)")
            ),
            hovertemplate="<b>Agent Role:</b> %{x}<br><b>Mean Duration:</b> %{y:.1f} seconds<extra></extra>"
        )
    ])
    fig.update_layout(
        title="Average Execution Duration by Agent Role",
        xaxis_title="Agent Role",
        yaxis_title="Mean Duration (Seconds)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_quality_by_topic(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive horizontal bar chart of mean quality score across topics.
    """
    if df.empty:
        return go.Figure()

    grouped = df.groupby("task_type")["quality_score"].mean().sort_values(ascending=True).reset_index()
    grouped.columns = ["task_type", "mean_quality"]

    fig = go.Figure(data=[
        go.Bar(
            y=grouped["task_type"],
            x=grouped["mean_quality"],
            orientation="h",
            marker=dict(
                color=grouped["mean_quality"],
                colorscale="Viridis",
                colorbar=dict(title="Score")
            ),
            text=[f"{v:.3f}" for v in grouped["mean_quality"]],
            textposition="inside",
            hovertemplate="<b>Research Topic:</b> %{y}<br><b>Mean Quality:</b> %{x:.3f}<extra></extra>"
        )
    ])
    fig.update_layout(
        title="Mean Quality Score Across Research Topics (0.0 to 1.0)",
        xaxis_title="Quality Score",
        yaxis_title="Research Topic",
        template="plotly_white",
        height=550,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_error_rate_by_agent(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive bar chart of simulated error rate (%) by agent role.
    """
    if df.empty:
        return go.Figure()

    grouped = df.groupby("agent_role")["errors"].agg(["count", "sum"]).reset_index()
    grouped["error_rate_pct"] = (grouped["sum"] / grouped["count"]) * 100

    fig = go.Figure(data=[
        go.Bar(
            x=grouped["agent_role"],
            y=grouped["error_rate_pct"],
            text=[f"{v:.2f}%" for v in grouped["error_rate_pct"]],
            textposition="auto",
            marker=dict(
                color=COLOR_DANGER,
                line=dict(width=1, color="rgba(0,0,0,0.2)")
            ),
            hovertemplate="<b>Agent Role:</b> %{x}<br><b>Error Rate:</b> %{y:.2f}%<extra></extra>"
        )
    ])
    fig.update_layout(
        title="Simulated Error Rate (%) by Agent Role",
        xaxis_title="Agent Role",
        yaxis_title="Fault Rate (%)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_token_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive histogram of tokens used with vertical line for mean.
    """
    if df.empty:
        return go.Figure()

    mean_val = df["tokens_used"].mean()

    fig = px.histogram(
        df,
        x="tokens_used",
        nbins=40,
        color="agent_role",
        color_discrete_map=AGENT_COLORS,
        barmode="overlay",
        opacity=0.75,
        title="Token Usage Distribution Across Benchmark Runs"
    )
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean: {mean_val:,.0f} tokens",
        annotation_position="top right"
    )
    fig.update_layout(
        xaxis_title="Tokens Used",
        yaxis_title="Frequency (Record Count)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend_title="Agent Role"
    )
    return fig
