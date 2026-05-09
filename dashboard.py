"""Streamlit dashboard — three-tab construction monitoring system.

Tabs:
  📊 Progress Tracking   — stage-by-stage mould progress (original pipeline)
  ⚡ Productivity Analytics — Lean/VA/NVA worker productivity (new pipeline)
  🦺 Safety Monitoring   — placeholder, coming soon
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import re
import csv
import io
import datetime
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from config import CLASS_NAMES, STAGE_COLORS, NUM_MOULDS, MOULD_CENTROIDS_FALLBACK as MOULD_CENTROIDS
from timestamp_ocr import timestamp_to_seconds, seconds_to_timestamp

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

from model_loader import ensure_models
ensure_models()

st.set_page_config(
    page_title="Casting Yard Monitor",
    page_icon="🏗️",
    layout="wide",
)

STAGE_ORDER = [
    "Mould Cleaning", "Rebar Cage Placement", "Concrete Pouring",
    "Surface Finishing", "Curing", "Vacuum Lifting",
]

# ── Primavera export helpers ──────────────────────────────────────────────────

def parse_duration_to_hours(dur_str: str) -> float:
    """Convert a human-readable duration string (e.g. '54m 30s') to decimal hours."""
    m = re.match(r"(\d+)m\s*(\d+)s", str(dur_str).strip())
    if m:
        return round((int(m.group(1)) * 60 + int(m.group(2))) / 3600, 4)
    m2 = re.match(r"(\d+)m", str(dur_str).strip())
    if m2:
        return round(int(m2.group(1)) / 60, 4)
    return 0.0


def parse_ocr_date(date_str: str) -> datetime.datetime:
    """
    Parse the OCR-extracted date string into a datetime object.

    Primary format from timestamp_ocr: 'DD MM YYYY'  e.g. '24 03 2026'
    Also handles '24/03/2026', '2026-03-24', '24-03-2026' as fallbacks.
    Returns today's date (midnight) if the string is missing or unparseable.
    """
    if not date_str:
        return datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    for fmt in ("%d %m %Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)


def build_primavera_xer(detail_df: pd.DataFrame, base_date: datetime.datetime, activity_id_map: dict) -> str:
    """
    Build a Primavera P6 spreadsheet-import compatible tab-delimited string.

    Columns produced (matching P6 Activity Import template):
        task_id | activity_id | activity_name | wbs_name |
        target_start_date | target_end_date | target_drtn_hr_cnt |
        act_start_date | act_end_date | status_code

    Datetime format used: '%d-%b-%y %H:%M'  e.g. '24-Mar-26 08:42'
    """
    def to_p6_dt(time_str: str) -> str:
        """Combine base_date with HH:MM:SS time string -> P6 datetime string."""
        try:
            h, mn, s = map(int, time_str.split(":"))
            dt = base_date + datetime.timedelta(hours=h, minutes=mn, seconds=s)
            return dt.strftime("%d-%b-%y %H:%M")
        except Exception:
            return base_date.strftime("%d-%b-%y %H:%M")

    rows = []
    task_counter = 1

    for _, row in detail_df.iterrows():
        stage = row["Stage"]
        mould = row["Mould"]
        seq   = row["#"]

        mapped_base = activity_id_map.get(stage, "")
        if mapped_base:
            act_id = f"{mapped_base}-{mould.replace(' ', '')}-{seq:02d}"
        else:
            act_id = f"{mould.replace(' ', '')}-{seq:02d}"

        dur_hrs  = parse_duration_to_hours(row["Duration"])
        start_p6 = to_p6_dt(row["Start"])
        end_p6   = to_p6_dt(row["End"])

        rows.append({
            "task_id":             f"T{task_counter:04d}",
            "activity_id":         act_id,
            "activity_name":       f"{mould} — {stage}",
            "wbs_name":            mould,
            "target_start_date":   start_p6,
            "target_end_date":     end_p6,
            "target_drtn_hr_cnt":  f"{dur_hrs:.4f}",
            "act_start_date":      start_p6,
            "act_end_date":        end_p6,
            "status_code":         "TK_Complete",
        })
        task_counter += 1

    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=rows[0].keys(),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def render_primavera_export(detail_df: pd.DataFrame, date_str: str):
    """
    Renders the Primavera P6 export UI below the existing CSV download button.

    date_str : raw OCR date string from metadata (e.g. '24 03 2026').
               Parsed internally — no date picker shown to the user.

    Session-state wizard:
      p6_step == 0  -> idle (trigger button shown)
      p6_step == 1  -> Activity ID mapping form -> generate & download
    """
    st.markdown("---")
    st.subheader("Export to Primavera P6")

    # Parse OCR date once — used for display, filename, and XER content
    base_date       = parse_ocr_date(date_str)
    date_human      = base_date.strftime("%d %B %Y")      # e.g. '24 March 2026'
    date_for_fname  = base_date.strftime("%Y-%m-%d")      # e.g. '2026-03-24'

    # ── Session state ─────────────────────────────────────────────────────────
    if "p6_step" not in st.session_state:
        st.session_state.p6_step = 0      # 0 = idle, 1 = mapping
    if "p6_act_map" not in st.session_state:
        st.session_state.p6_act_map = {}

    # ── Step 0 — idle: show trigger button + date badge ───────────────────────
    if st.session_state.p6_step == 0:
        col_btn, col_date = st.columns([2, 3])
        with col_btn:
            if st.button("Generate Primavera-Compatible File", use_container_width=True):
                st.session_state.p6_step = 1
                st.rerun()
        with col_date:
            if date_str:
                st.info(f"Recording date from OCR: **{date_human}**")
            else:
                st.warning("No date detected by OCR — today's date will be used.")
        return

    # ── Step 1 — Activity ID mapping ─────────────────────────────────────────
    # Show which date will be embedded in the XER file (read-only, from OCR)
    if date_str:
        st.success(f"Using OCR date: **{date_human}**")
    else:
        st.warning(
            f"No OCR date found — using today's date ({date_human}) instead."
        )

    st.markdown("**Activity ID Mapping — Primavera Project**")
    st.caption(
        "Enter the Activity ID prefix already defined in your Primavera project "
        "for each construction stage. Leave blank to auto-generate IDs. "
        "A unique suffix (Mould + sequence number) will be appended automatically."
    )

    unique_stages = detail_df["Stage"].unique().tolist()
    act_map = {}

    cols = st.columns(2)
    for i, stage in enumerate(unique_stages):
        with cols[i % 2]:
            default = st.session_state.p6_act_map.get(stage, "")
            act_map[stage] = st.text_input(
                label=stage,
                value=default,
                placeholder=f"e.g. A{1000 + i * 10}",
                key=f"p6_act_{stage}",
            )

    st.caption(
        "**Preview of generated Activity IDs** "
        "(suffix = Mould identifier + sequence number):"
    )
    preview_rows = []
    for stage, base_id in act_map.items():
        full_id = f"{base_id}-Mould1-01" if base_id else "Mould1-01"
        preview_rows.append({
            "Stage":            stage,
            "Your Activity ID": base_id or "(auto)",
            "Example full ID":  full_id,
        })
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    col_gen, col_cancel = st.columns([2, 1])
    with col_gen:
        generate = st.button(
            "Generate & Download XER File", type="primary", key="p6_generate"
        )
    with col_cancel:
        if st.button("Cancel", key="p6_cancel"):
            st.session_state.p6_step = 0
            st.session_state.p6_act_map = {}
            st.rerun()

    if generate:
        st.session_state.p6_act_map = act_map
        xer_content = build_primavera_xer(detail_df, base_date, act_map)
        if xer_content:
            fname = f"primavera_import_{date_for_fname}.xer"
            st.success(
                f"File ready — **{len(detail_df)} activities** mapped across "
                f"{detail_df['Mould'].nunique()} moulds for **{date_human}**."
            )
            st.info(
                "**How to import into Primavera P6:**  \n"
                "File → Import → Spreadsheet (XLS/XLSX/Tab-delimited) → "
                "select *Activities* as the import type → map columns to "
                "`activity_id`, `activity_name`, `target_start_date`, etc.",
                icon="ℹ️",
            )
            st.download_button(
                label="Download Primavera XER File",
                data=xer_content,
                file_name=fname,
                mime="text/tab-separated-values",
                use_container_width=True,
                type="primary",
            )
            if st.button("Start Over", key="p6_reset"):
                st.session_state.p6_step = 0
                st.session_state.p6_act_map = {}
                st.rerun()
        else:
            st.error("No data to export. Please check the progress report.")


# ── Progress helper functions ─────────────────────────────────────────────────

def compute_segment_counts(timelines):
    segment_counts  = {}
    current_cycle_stages = {}
    for mould_name, stages in timelines.items():
        count, current = 0, set()
        for s in stages:
            current.add(s["stage"])
            if s["stage"] == "Vacuum Lifting":
                count += 1
                current = set()
        segment_counts[mould_name]       = count
        current_cycle_stages[mould_name] = current
    return segment_counts, current_cycle_stages


def build_gantt_chart(timelines):
    rows = []
    for mould_name, stages in timelines.items():
        for s in stages:
            start_sec = timestamp_to_seconds(s["start"])
            end_sec   = timestamp_to_seconds(s["end"])
            if end_sec <= start_sec:
                end_sec = start_sec + 1
            rows.append({
                "Mould": mould_name, "Stage": s["stage"],
                "Start": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=start_sec),
                "End":   pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=end_sec),
                "Start Time": s["start"], "End Time": s["end"],
                "Duration (min)": round((end_sec - start_sec) / 60, 1),
            })
    if not rows:
        return None, None
    df    = pd.DataFrame(rows)
    x_min = df["Start"].min()
    x_max = df["End"].max()
    fig   = px.timeline(
        df, x_start="Start", x_end="End", y="Mould", color="Stage",
        color_discrete_map=STAGE_COLORS,
        hover_data=["Start Time", "End Time", "Duration (min)"],
    )
    fig.update_layout(
        xaxis_title="Time", yaxis_title="", height=250,
        legend_title="Stage", margin=dict(t=30, b=30),
        xaxis=dict(tickformat="%H:%M:%S", range=[x_min, x_max]),
    )
    fig.update_yaxes(autorange="reversed")
    return fig, df


def build_stage_duration_chart(df):
    if df is None or df.empty:
        return None
    agg = df.groupby(["Mould", "Stage"])["Duration (min)"].sum().reset_index()
    fig = px.bar(
        agg, x="Mould", y="Duration (min)", color="Stage",
        color_discrete_map=STAGE_COLORS, barmode="group",
        title="Stage Duration per Mould (minutes)",
    )
    fig.update_layout(height=350, margin=dict(t=40, b=30))
    return fig


def build_completion_matrix(timelines):
    _, current_cycle_stages = compute_segment_counts(timelines)
    prog_data = []
    for mould_name in timelines:
        done = current_cycle_stages[mould_name]
        for stage in STAGE_ORDER:
            prog_data.append({
                "Mould": mould_name, "Stage": stage,
                "Completed": "Yes" if stage in done else "No",
            })
    if not prog_data:
        return None
    prog_df = pd.DataFrame(prog_data)
    prog_df["Stage"] = pd.Categorical(prog_df["Stage"], categories=STAGE_ORDER, ordered=True)
    fig = px.scatter(
        prog_df, x="Stage", y="Mould", color="Completed",
        color_discrete_map={"Yes": "#2ecc71", "No": "#e0e0e0"},
        symbol="Completed", symbol_map={"Yes": "circle", "No": "circle-open"},
        title="Stage Completion Matrix (Current Cycle)",
    )
    fig.update_traces(marker_size=20)
    fig.update_layout(
        height=280, margin=dict(t=40),
        xaxis=dict(categoryorder="array", categoryarray=STAGE_ORDER),
    )
    return fig


def build_segment_count_display(timelines):
    segment_counts, _ = compute_segment_counts(timelines)
    cols = st.columns(len(timelines))
    for i, (mould_name, count) in enumerate(segment_counts.items()):
        with cols[i]:
            st.metric(mould_name, f"{count} Segment{'s' if count != 1 else ''}")


def build_detailed_table(timelines):
    rows = []
    for mould_name, stages in timelines.items():
        for idx, s in enumerate(stages, 1):
            s_sec = timestamp_to_seconds(s["start"])
            e_sec = timestamp_to_seconds(s["end"])
            rows.append({
                "Mould": mould_name, "#": idx, "Stage": s["stage"],
                "Start": s["start"], "End": s["end"],
                "Duration": f"{(e_sec - s_sec) // 60}m {(e_sec - s_sec) % 60}s",
            })
    return pd.DataFrame(rows)


def load_results(json_path="output/progress_report.json"):
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def get_interim_timelines(tracker):
    timelines_raw = tracker.build_timeline()
    summary = {}
    for mould_idx in range(tracker.num_moulds):
        mould_name = f"Mould {mould_idx + 1}"
        stages = timelines_raw.get(mould_idx, [])
        summary[mould_name] = [
            {"stage": s["stage"], "start": s["start"], "end": s["end"]}
            for s in stages
        ]
    return summary


def render_last_mould_status(timelines):
    cols = st.columns(NUM_MOULDS)
    for i, (mould_name, stages) in enumerate(timelines.items()):
        with cols[i]:
            if stages:
                st.metric(mould_name, stages[-1]["stage"], f"Since {stages[-1]['start']}")
            else:
                st.metric(mould_name, "No Data", "--")


def render_report_body(timelines, metadata):
    date_str = metadata.get("date")
    if date_str:
        st.markdown(f"**Date:** {date_str}")

    st.subheader("Latest Mould Status")
    render_last_mould_status(timelines)

    st.subheader("Segments Completed (Mould-wise)")
    build_segment_count_display(timelines)

    with st.expander("Processing Metadata", expanded=False):
        mc = st.columns(4)
        mc[0].metric("Frames Processed", f"{metadata['frames_processed']:,}")
        mc[1].metric("Resolution", metadata["resolution"])
        mc[2].metric("FPS", f"{metadata['fps']:.1f}")
        mc[3].metric("OCR Failures", metadata["ocr_failures"])
        centroids = metadata.get("mould_centroids")
        if centroids:
            st.markdown("**Auto-detected Mould Centroids:**")
            c_cols = st.columns(len(centroids))
            for i, (k, v) in enumerate(sorted(centroids.items())):
                c_cols[i].markdown(f"Mould {int(k)+1}: `({v[0]:.3f}, {v[1]:.3f})`")

    st.subheader("Activity Timeline")
    gantt_fig, gantt_df = build_gantt_chart(timelines)
    if gantt_fig:
        st.plotly_chart(gantt_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        dur_fig = build_stage_duration_chart(gantt_df)
        if dur_fig:
            st.plotly_chart(dur_fig, use_container_width=True)
    with c2:
        comp_fig = build_completion_matrix(timelines)
        if comp_fig:
            st.plotly_chart(comp_fig, use_container_width=True)

    st.subheader("Detailed Progress Report")
    detail_df = build_detailed_table(timelines)
    if not detail_df.empty:
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Report (CSV)", detail_df.to_csv(index=False),
            "mould_progress_report.csv", "text/csv",
        )
        # ── Primavera export — date sourced from OCR, no picker needed ────────
        render_primavera_export(detail_df, date_str)

    st.subheader("Per-Mould Breakdown")
    tabs = st.tabs([f"Mould {i+1}" for i in range(len(timelines))])
    for i, (mould_name, stages) in enumerate(timelines.items()):
        with tabs[i]:
            if not stages:
                st.info(f"No activity detected for {mould_name}.")
                continue
            for s in stages:
                s_sec   = timestamp_to_seconds(s["start"])
                e_sec   = timestamp_to_seconds(s["end"])
                dur_min = (e_sec - s_sec) / 60
                color   = STAGE_COLORS.get(s["stage"], "#888")
                st.markdown(
                    f'<div style="padding:8px;margin:4px 0;border-left:4px solid {color};'
                    f'background:#f8f9fa;border-radius:4px;">'
                    f'<strong>{s["stage"]}</strong> &mdash; '
                    f'{s["start"]} to {s["end"]} <em>({dur_min:.1f} min)</em></div>',
                    unsafe_allow_html=True,
                )


# ── Productivity helpers ──────────────────────────────────────────────────────

_PROD_REPORT_PATH    = "output/productivity_report.json"
_PROD_HEATMAP_PATH   = "output/productivity_heatmap.png"
_PROD_SPAGHETTI_PATH = "output/productivity_spaghetti.png"

_SAFETY_REPORT_PATH  = "output/safety_report.json"


def extract_boundary_points_from_canvas(json_data, scale_x, scale_y):
    """Pull boundary points from streamlit-drawable-canvas JSON and scale to frame size."""
    pts = []
    if not json_data:
        return pts
    for obj in json_data.get("objects", []):
        if obj.get("type") == "circle":
            radius = obj.get("radius", 5)
            px = obj.get("left", 0) + radius
            py = obj.get("top",  0) + radius
            pts.append((int(px * scale_x), int(py * scale_y)))
    pts.sort(key=lambda p: p[0])
    return pts


def save_productivity_results(result):
    """Persist productivity results to disk (mirrors progress tab's JSON approach)."""
    from productivity_processor import build_heatmap_overlay, build_spaghetti_diagram
    os.makedirs("output", exist_ok=True)
    tl = result["timeline_history"]
    report = {
        "timeline_history":   {str(k): v for k, v in tl.items()},
        "frame_count":        result["frame_count"],
        "frame_skip":         result["frame_skip"],
        "timelapse_interval": result["timelapse_interval"],
        "unique_workers":     len(result["worker_trajectories"]),
    }
    with open(_PROD_REPORT_PATH, "w") as f:
        json.dump(report, f)
    heatmap_rgb = build_heatmap_overlay(result["heatmap_data"], result["first_frame"])
    Image.fromarray(heatmap_rgb).save(_PROD_HEATMAP_PATH)
    if result["worker_trajectories"]:
        spag_rgb = build_spaghetti_diagram(result["worker_trajectories"], result["first_frame"])
        Image.fromarray(spag_rgb).save(_PROD_SPAGHETTI_PATH)


def load_productivity_results():
    """Load saved productivity results from disk. Returns None if not found."""
    if not os.path.exists(_PROD_REPORT_PATH):
        return None
    with open(_PROD_REPORT_PATH) as f:
        data = json.load(f)
    data["timeline_history"] = {int(k): v for k, v in data["timeline_history"].items()}
    return data


def render_productivity_report_body(data):
    """Render the full productivity report — callable from both live and saved-data paths."""
    from collections import Counter
    from productivity_processor import (
        build_crew_balance_chart, build_lean_pie_charts,
        compute_lean_totals, LEAN_CATEGORY_MAP,
    )
    tl                      = data["timeline_history"]
    frame_skip              = data["frame_skip"]
    timelapse_interval      = data["timelapse_interval"]
    unique_workers          = data.get("unique_workers", 0)
    tl_totals, tl_per_mould = compute_lean_totals(tl)
    total_tracked           = sum(tl_totals.values()) or 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Value-Adding (VA)",       f"{tl_totals['VA']   / total_tracked * 100:.1f}%")
    m2.metric("Necessary NVA (NVAN)",    f"{tl_totals['NVAN'] / total_tracked * 100:.1f}%")
    m3.metric("Non-Value Adding (NVA)",  f"{tl_totals['NVA']  / total_tracked * 100:.1f}%")
    m4.metric("Unique Workers Tracked",  str(unique_workers))

    st.subheader("Crew Balance Timeline")
    crew_fig = build_crew_balance_chart(tl, frame_skip, timelapse_interval)
    if crew_fig:
        st.plotly_chart(crew_fig, use_container_width=True)

    st.subheader("Lean Activity Distribution")
    pie_fig = build_lean_pie_charts(tl)
    if pie_fig:
        st.plotly_chart(pie_fig, use_container_width=True)

    st.subheader("Per-Mould Lean Summary")
    lean_rows = []
    for mould_id, counts in tl_per_mould.items():
        total_m = sum(counts.values()) or 1
        lean_rows.append({
            "Mould":         f"Mould {mould_id}",
            "VA %":          f"{counts['VA']   / total_m * 100:.1f}%",
            "NVAN %":        f"{counts['NVAN'] / total_m * 100:.1f}%",
            "NVA %":         f"{counts['NVA']  / total_m * 100:.1f}%",
            "VA Frames":     counts["VA"],
            "NVAN Frames":   counts["NVAN"],
            "NVA Frames":    counts["NVA"],
        })
    if lean_rows:
        st.dataframe(pd.DataFrame(lean_rows), use_container_width=True, hide_index=True)

    c_heat, c_spag = st.columns(2)
    with c_heat:
        st.subheader("Motion Heatmap")
        if os.path.exists(_PROD_HEATMAP_PATH):
            st.image(_PROD_HEATMAP_PATH, caption="Worker movement concentration",
                     use_column_width=True)
        else:
            st.info("Heatmap image not found.")
    with c_spag:
        st.subheader("Spaghetti Diagram")
        if os.path.exists(_PROD_SPAGHETTI_PATH):
            st.image(_PROD_SPAGHETTI_PATH, caption="Worker movement paths (colour per ID)",
                     use_column_width=True)
        else:
            st.info("No trajectory data recorded.")

    st.subheader("Phase Breakdown (per Mould)")
    if tl:
        mould_tabs = st.tabs([f"Mould {mid}" for mid in tl.keys()])
        for i, (mould_id, history) in enumerate(tl.items()):
            with mould_tabs[i]:
                if not history:
                    st.info("No data.")
                    continue
                phase_counts = Counter(history)
                rows_ph = []
                for phase, cnt in sorted(phase_counts.items(), key=lambda x: -x[1]):
                    real_min = cnt * frame_skip * timelapse_interval / 60.0
                    cat = LEAN_CATEGORY_MAP.get(phase, "NVA")
                    rows_ph.append({
                        "Phase": phase, "Lean": cat,
                        "Frames": cnt,
                        "Real-World Time (min)": round(real_min, 2),
                    })
                st.dataframe(pd.DataFrame(rows_ph), use_container_width=True, hide_index=True)


# ── Safety helpers ───────────────────────────────────────────────────────────

def save_safety_results(result):
    os.makedirs("output", exist_ok=True)
    report = {k: result.get(k, v) for k, v in {
        "danger_events":       0,
        "warning_events":      0,
        "total_no_ppe":        0,
        "total_no_helmet":     0,
        "total_no_vest":       0,
        "frame_count":         0,
        "event_log":           [],
        "fps":                 25,
        "ppe_compliance_pct":  0,
        "closest_approach":    None,
        "closest_time":        "—",
        "proximity_timeline":  [],
        "workers_per_minute":  {},
        "caution_per_minute":  {},
        "warning_per_minute":  {},
        "cum_caution_workers": 0,
        "cum_warning_workers": 0,
        "cum_safe_workers":    0,
        "safety_index":        0,
        "robust_closest":      None,
        "dist_median":         None,
        "dist_mean":           None,
        "ppm":                       0,
        "scale_y":                   1.0,
        "cam_type":                  "—",
        "video_res":                 "—",
        "cam_is_tlc":                False,
        "ppe_compliance_timeline":   [],
        "best_frame_workers":        0,
        "max_no_ppe_count":          0,
        "max_caution_count":         0,
        "total_dwell_caution_sec":   0,
        "total_dwell_warning_sec":   0,
        "best_frame_img_path":       None,
        "zone_boundary_img_path":    None,
        "no_ppe_frame_img_path":     None,
        "caution_frame_img_path":    None,
        "zone_count_timeline":       [],
        "aerolift_active_sec":       0,
        "aerolift_active_pct":       0,
    }.items()}
    with open(_SAFETY_REPORT_PATH, "w") as f:
        json.dump(report, f)


def load_safety_results():
    if not os.path.exists(_SAFETY_REPORT_PATH):
        return None
    with open(_SAFETY_REPORT_PATH) as f:
        return json.load(f)


def render_safety_report_body(data):
    d          = data.get("danger_events", 0)
    w          = data.get("warning_events", 0)
    np_        = data.get("total_no_ppe", 0)
    ppe_pct    = data.get("ppe_compliance_pct", 0)
    closest    = data.get("closest_approach", None)
    close_t    = data.get("closest_time", "—")
    prox_tl    = data.get("proximity_timeline", [])
    wpm        = data.get("workers_per_minute", {})
    cpm        = data.get("caution_per_minute", {})
    warnpm     = data.get("warning_per_minute", {})
    cum_c      = data.get("cum_caution_workers", 0)
    cum_w      = data.get("cum_warning_workers", 0)
    cum_s      = data.get("cum_safe_workers", 0)
    risk       = data.get("safety_index", data.get("hazard_score", data.get("risk_score", 0)))
    robust_cl  = data.get("robust_closest", None)
    dist_median= data.get("dist_median", None)
    dist_mean  = data.get("dist_mean", None)
    fc         = data.get("frame_count", 0)
    fps_val    = data.get("fps", 25)
    ppm        = data.get("ppm", 0)
    scale_y    = data.get("scale_y", 1.0)
    cam_type   = data.get("cam_type", "—")
    video_res  = data.get("video_res", "—")
    event_log  = data.get("event_log", [])
    closest_str  = f"{closest*100:.0f} cm" if (closest and closest > 0.05) else "—"
    duration_min = round(fc / max(fps_val, 1) / 60, 1)
    cam_is_tlc           = data.get("cam_is_tlc", False)
    ppe_tl               = data.get("ppe_compliance_timeline", [])
    dwell_c_sec          = data.get("total_dwell_caution_sec", 0)
    dwell_w_sec          = data.get("total_dwell_warning_sec", 0)
    best_workers         = data.get("best_frame_workers", 0)
    max_no_ppe_count     = data.get("max_no_ppe_count", 0)
    max_caution_count    = data.get("max_caution_count", 0)
    best_img_path        = data.get("best_frame_img_path", None)
    zone_img_path        = data.get("zone_boundary_img_path", None)
    no_ppe_img_path      = data.get("no_ppe_frame_img_path", None)
    caution_img_path     = data.get("caution_frame_img_path", None)
    zone_count_tl        = data.get("zone_count_timeline", [])
    aerolift_sec         = data.get("aerolift_active_sec", 0)
    aerolift_pct         = data.get("aerolift_active_pct", 0)

    # ── Risk score banner ──────────────────────────────────────────────────────
    risk_col = "#22cc55" if risk < 30 else "#ff9900" if risk < 70 else "#cc0000"
    risk_lbl = "LOW SAFETY INDEX " if risk < 30 else "MEDIUM SAFETY INDEX" if risk < 70 else "HIGH SAFETY INDEX"
    st.markdown(
        f'<div style="padding:14px;border-radius:8px;background:{risk_col}22;'
        f'border:2px solid {risk_col};text-align:center;margin-bottom:16px;">'
        f'<span style="font-size:13px;color:{risk_col};">SITE SAFETY INDEX</span><br/>'
        f'<span style="font-size:40px;font-weight:bold;color:{risk_col};">{risk}/100</span>'
        f'&nbsp;&nbsp;<span style="font-size:18px;color:{risk_col};">{risk_lbl}</span></div>',
        unsafe_allow_html=True)

    # ── Top metrics ───────────────────────────────────────────────────────────
    true_min   = closest if (closest and closest > 0.20) else None
    show_str_m = f"{true_min*100:.0f} cm" if true_min else "—"

    cau_col_hex  = "#ffffff"
    wrn_col_hex  = "#ffffff"
    nppe_col_hex = "#ffffff"
    ppe_col_hex  = "#ffffff"

    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:8px;">'
        f'<div style="background:#1a1a1a;border-radius:8px;padding:16px;text-align:center;border-top:3px solid {cau_col_hex};">'
        f'<div style="font-size:12px;color:#aaa;margin-bottom:6px;">Caution Events</div>'
        f'<div style="font-size:32px;font-weight:bold;color:{cau_col_hex};">{d}</div>'
        f'</div>'
        f'<div style="background:#1a1a1a;border-radius:8px;padding:16px;text-align:center;border-top:3px solid {wrn_col_hex};">'
        f'<div style="font-size:12px;color:#aaa;margin-bottom:6px;">Warning Events</div>'
        f'<div style="font-size:32px;font-weight:bold;color:{wrn_col_hex};">{w}</div>'
        f'</div>'
        f'<div style="background:#1a1a1a;border-radius:8px;padding:16px;text-align:center;border-top:3px solid {nppe_col_hex};">'
        f'<div style="font-size:12px;color:#aaa;margin-bottom:6px;">No PPE</div>'
        f'<div style="font-size:32px;font-weight:bold;color:{nppe_col_hex};">{np_}</div>'
        f'</div>'
        f'<div style="background:#1a1a1a;border-radius:8px;padding:16px;text-align:center;border-top:3px solid {ppe_col_hex};">'
        f'<div style="font-size:12px;color:#aaa;margin-bottom:6px;">PPE Compliance</div>'
        f'<div style="font-size:32px;font-weight:bold;color:{ppe_col_hex};">{ppe_pct}%</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Calibration info ──────────────────────────────────────────────────────
    st.subheader("Calibration & Video Info")
    ci1, ci2, ci3, ci4 = st.columns(4)
    _cam_short = "Camera 1" if "Camera 1" in cam_type else ("Camera 2" if "Camera 2" in cam_type else cam_type.split("(")[0].strip())
    ci1.metric("Camera",      _cam_short)
    ci2.metric("Resolution",  video_res)
    ci3.metric("Calibration", f"{ppm} px/m")
    ci4.metric("Scale Y",     str(scale_y))
    st.caption(f"Camera: {cam_type} | Duration: {duration_min} min | FPS: {fps_val:.1f}")

    _al_col1, _al_col2, _al_col3 = st.columns(3)
    _al_col1.metric("Duration Analysed", f"{duration_min} min")
    _al_col2.metric("Aerolift Active Time", f"{aerolift_sec} s ({round(aerolift_sec/60,1)} min)")
    _al_col3.metric("Aerolift Active %", f"{aerolift_pct}%")

    st.markdown("---")

    # ── Camera Detection Mode ─────────────────────────────────────────────────
    st.subheader("Camera Detection Mode")
    if cam_is_tlc:
        st.markdown(
            '<div style="padding:12px;border-radius:8px;background:#0d1a2e;border-left:4px solid #4488ff;">'
            '<strong style="color:#4488ff;">TLC Overhead Camera</strong><br/>'
            '<span style="color:#aaa;font-size:13px;">'
            'Uses <strong>yolov8m only</strong> at confidence 0.55. '
            'PPE model and proximity model person sources are disabled to avoid '
            'false positives from structural elements visible in the overhead angle.'
            '</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="padding:12px;border-radius:8px;background:#0d1a2e;border-left:4px solid #22cc55;">'
            '<strong style="color:#22cc55;">Aerolift Camera</strong><br/>'
            '<span style="color:#aaa;font-size:13px;">'
            'Uses <strong>4 combined sources</strong>: '
            '(1) yolov8m full frame, (2) yolov8m top-half crop, '
            '(3) PPE model worker class, (4) Proximity model person class. '
            'All sources deduped with IoU 0.45.'
            '</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Key Detection Frames ──────────────────────────────────────────────────
    st.subheader("Key Detection Frames")

    _r1c1, _r1c2 = st.columns(2)
    with _r1c1:
        st.markdown("**Zone Boundary**")
        st.caption("First frame with aerolift zones drawn — Red <0.6m  Orange 0.6-2m  Green 2-4m")
        if zone_img_path and os.path.exists(zone_img_path):
            st.image(zone_img_path, use_column_width=True)
        else:
            st.info("No aerolift detected.")
    with _r1c2:
        st.markdown(f"**Peak Workers — {best_workers} detected**")
        st.caption("Frame with most simultaneous workers in work zone")
        if best_img_path and os.path.exists(best_img_path):
            st.image(best_img_path, use_column_width=True)
        else:
            st.info("Not available.")

    st.markdown("")

    _r2c1, _r2c2 = st.columns(2)
    with _r2c1:
        st.markdown(f"**Max No-PPE — {max_no_ppe_count} workers without PPE**")
        st.caption("Frame where the most workers were simultaneously detected without PPE")
        if no_ppe_img_path and os.path.exists(no_ppe_img_path):
            st.image(no_ppe_img_path, use_column_width=True)
        else:
            st.info("No PPE violations detected." if max_no_ppe_count == 0
                    else "Image not available.")
    with _r2c2:
        st.markdown(f"**Max Caution — {max_caution_count} workers in caution zone**")
        st.caption("Frame where the most workers were simultaneously inside 0.6m of aerolift")
        if caution_img_path and os.path.exists(caution_img_path):
            st.image(caution_img_path, use_column_width=True)
        else:
            st.info("No caution events detected." if max_caution_count == 0
                    else "Image not available.")

    st.markdown("---")

    # ── PPE Compliance Over Time ──────────────────────────────────────────────
    st.subheader("PPE Compliance Over Time")
    if ppe_tl:
        _t   = [p["t"]   for p in ppe_tl]
        _pct = [p["pct"] for p in ppe_tl]
        fig_ppe_t = go.Figure()
        fig_ppe_t.add_trace(go.Scatter(
            x=_t, y=_pct, mode="lines",
            line=dict(color="#22cc55", width=2),
            fill="tozeroy", fillcolor="rgba(34,204,85,0.06)"))
        fig_ppe_t.add_hline(y=80, line_dash="dot", line_color="#ff9900",
                            annotation_text="80% target", annotation_font_color="#ff9900")
        fig_ppe_t.update_layout(
            xaxis_title="Time (s)", yaxis_title="Compliance %",
            yaxis=dict(range=[0, 105]), height=240,
            margin=dict(t=20, b=30, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color="#ccc"), showlegend=False)
        st.plotly_chart(fig_ppe_t, use_container_width=True)
        st.caption("Drops below 80% indicate periods workers removed or lost PPE.")
    else:
        st.info("PPE compliance timeline not available.")

    st.markdown("---")

    # ── Worker Dwell Time ─────────────────────────────────────────────────────
    st.subheader("Worker Dwell Time in Proximity Zones")
    _dw1, _dw2 = st.columns(2)
    with _dw1:
        _dc_col = "#cc0000" if dwell_c_sec > 30 else "#ff9900" if dwell_c_sec > 10 else "#22cc55"
        st.markdown(
            f'<div style="padding:12px;border-radius:8px;background:#1a0000;border-left:4px solid {_dc_col};">'
            f'<div style="font-size:12px;color:#aaa;">Total time in Caution Zone (&lt;0.6m)</div>'
            f'<div style="font-size:28px;font-weight:bold;color:{_dc_col};">{dwell_c_sec} s</div>'
            f'<div style="font-size:12px;color:#888;">({round(dwell_c_sec/60,1)} min)</div>'
            f'<div style="font-size:11px;color:#666;margin-top:4px;">Cumulative seconds all workers spent within 0.6m of aerolift</div>'
            f'</div>', unsafe_allow_html=True)
    with _dw2:
        _dw_col = "#ff9900" if dwell_w_sec > 60 else "#22cc55"
        st.markdown(
            f'<div style="padding:12px;border-radius:8px;background:#1a1000;border-left:4px solid {_dw_col};">'
            f'<div style="font-size:12px;color:#aaa;">Total time in Warning Zone (0.6-2m)</div>'
            f'<div style="font-size:28px;font-weight:bold;color:{_dw_col};">{dwell_w_sec} s</div>'
            f'<div style="font-size:12px;color:#888;">({round(dwell_w_sec/60,1)} min)</div>'
            f'<div style="font-size:11px;color:#666;margin-top:4px;">Cumulative seconds all workers spent within 0.6-2m of aerolift</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Zone breakdown pie chart + bar ────────────────────────────────────────
    col_pie, col_zone = st.columns(2)
    with col_pie:
        st.subheader("Worker Zone Distribution")
        total_zone = cum_c + cum_w + cum_s
        if total_zone > 0:
            fig_pie = go.Figure(go.Pie(
                labels=["Caution Zone", "Warning Zone", "Safe Zone"],
                values=[cum_c, cum_w, cum_s],
                marker=dict(colors=["#cc0000", "#ff9900", "#22cc55"]),
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                height=280, margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"),
                showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No zone data recorded.")

    with col_zone:
        st.subheader("Zone Event Counts")
        if total_zone > 0:
            fig_bar = go.Figure(go.Bar(
                x=["Caution", "Warning", "Safe"],
                y=[cum_c, cum_w, cum_s],
                marker_color=["#cc0000", "#ff9900", "#22cc55"],
                text=[cum_c, cum_w, cum_s],
                textposition="outside"))
            fig_bar.update_layout(
                height=280, margin=dict(t=20, b=30, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.05)",
                font=dict(color="#ccc"), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No zone data recorded.")

    # ── PPE compliance gauge ───────────────────────────────────────────────────
    col_ppe, col_close = st.columns(2)
    with col_ppe:
        st.subheader("PPE Compliance Rate")
        bar_col = "#22cc55" if ppe_pct>=80 else "#ff9900" if ppe_pct>=50 else "#cc0000"
        st.markdown(
            f'<div style="background:#333;border-radius:6px;height:22px;margin-bottom:6px;">'
            f'<div style="background:{bar_col};width:{min(100,int(ppe_pct))}%;'
            f'height:22px;border-radius:6px;"></div></div>'
            f'<div style="font-size:30px;font-weight:bold;color:{bar_col};">{ppe_pct}%</div>',
            unsafe_allow_html=True)
        st.caption("Helmet OR vest detected = compliant. Workers too small to judge = excluded.")

    with col_close:
        st.subheader("Proximity to Aerolift")
        true_min_r = closest if (closest and closest > 0.20) else None
        true_min_str = f"{true_min_r*100:.0f} cm" if true_min_r else "—"
        med_show = dist_median if (dist_median and dist_median > 0.20) else None
        med_str2 = f"{med_show*100:.0f} cm" if med_show else "—"
        cl_col   = "#cc0000" if (true_min_r and true_min_r<0.6) else "#ff9900" if (true_min_r and true_min_r<2.0) else "#22cc55"
        gauge_val   = med_show if med_show else 4.0
        gauge_pct   = min(100, int((gauge_val / 4.0) * 100))
        caution_w   = 15
        warning_w   = 35
        safe_w      = 50
        filled      = gauge_pct
        c_fill      = min(filled, caution_w)
        w_fill      = min(max(filled - caution_w, 0), warning_w)
        s_fill      = max(filled - caution_w - warning_w, 0)
        st.markdown(
            f'<div style="padding:8px;border-radius:6px;background:#1a0a0a;border-left:3px solid {cl_col};margin-bottom:8px;">'
            f'<div style="font-size:10px;color:#aaa;">Closest Ever Recorded</div>'
            f'<div style="font-size:26px;font-weight:bold;color:{cl_col};">{true_min_str}</div>'
            f'<div style="font-size:10px;color:#666;">at {close_t}</div></div>'
            f'<div style="font-size:10px;color:#aaa;margin-bottom:4px;">Typical distance gauge  (0m = caution  2m = warning  4m = safe)</div>'
            f'<div style="display:flex;gap:2px;height:12px;border-radius:4px;overflow:hidden;margin-bottom:4px;">'
            f'  <div style="width:{caution_w}%;background:#3a0000;border-radius:3px 0 0 3px;overflow:hidden;">'
            f'    <div style="width:{int(c_fill/caution_w*100) if caution_w else 0}%;height:100%;background:#cc0000;"></div></div>'
            f'  <div style="width:{warning_w}%;background:#2a1800;overflow:hidden;">'
            f'    <div style="width:{int(w_fill/warning_w*100) if warning_w else 0}%;height:100%;background:#ff9900;"></div></div>'
            f'  <div style="width:{safe_w}%;background:#001a00;border-radius:0 3px 3px 0;overflow:hidden;">'
            f'    <div style="width:{int(s_fill/safe_w*100) if safe_w else 0}%;height:100%;background:#22cc55;"></div></div>'
            f'</div>'
            f'<div style="position:relative;height:18px;margin-top:4px;font-size:10px;">'
            f'  <span style="position:absolute;left:0%;color:#cc0000;">0m</span>'
            f'  <span style="position:absolute;left:15%;transform:translateX(-50%);color:#ff9900;">0.6m</span>'
            f'  <span style="position:absolute;left:50%;transform:translateX(-50%);color:#22cc55;">2m</span>'
            f'  <span style="position:absolute;right:0%;color:#22cc55;">4m</span>'
            f'</div>',
            unsafe_allow_html=True)
        if true_min_r and true_min_r < 0.60:
            st.error("Worker entered caution zone — review event log for details.")
        elif true_min_r and true_min_r < 2.0:
            st.markdown(
                f'<div style="padding:10px 14px;border-radius:6px;background:#1f1200;'
                f'border:1px solid #ff9900;border-left:4px solid #ff9900;margin-top:6px;">'
                f'<span style="color:#ff9900;font-size:14px;">&#9888; Workers came within warning zone during this session.</span>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Zone person count timeline ───────────────────────────────────────────
    st.subheader("Workers in Each Zone — Over Time")
    if zone_count_tl:
        _zt  = [p["t"]       for p in zone_count_tl]
        _zc  = [p["caution"] for p in zone_count_tl]
        _zw  = [p["warning"] for p in zone_count_tl]
        _zs  = [p.get("safe",0) for p in zone_count_tl]
        fig_zone = go.Figure()
        fig_zone.add_trace(go.Scatter(
            x=_zt, y=_zc, mode="lines", name="Caution",
            line=dict(color="#cc0000", width=2),
            fill="tozeroy", fillcolor="rgba(204,0,0,0.08)"))
        fig_zone.add_trace(go.Scatter(
            x=_zt, y=_zw, mode="lines", name="Warning",
            line=dict(color="#ff9900", width=2),
            fill="tozeroy", fillcolor="rgba(255,153,0,0.06)"))
        fig_zone.add_trace(go.Scatter(
            x=_zt, y=_zs, mode="lines", name="Safe",
            line=dict(color="#22cc55", width=1.5),
            fill="tozeroy", fillcolor="rgba(34,204,85,0.04)"))
        fig_zone.update_layout(
            xaxis_title="Time (s)", yaxis_title="Number of Workers",
            yaxis=dict(rangemode="nonnegative"),
            height=260, margin=dict(t=20, b=30, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color="#ccc"),
            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_zone, use_container_width=True)
        st.caption("Number of workers simultaneously in each proximity zone. "
                   "Peaks in red/orange indicate high-risk periods.")
        _zdf = pd.DataFrame(zone_count_tl).rename(columns={
            "t":"Time (s)","caution":"Caution Workers",
            "warning":"Warning Workers","safe":"Safe Workers","total":"Total Workers"})
        st.download_button("Zone Count Timeline (CSV)", _zdf.to_csv(index=False),
                           "zone_count_timeline.csv", "text/csv")
    else:
        st.info("Zone count timeline not available.")

    st.markdown("---")

    # ── Proximity timeline ────────────────────────────────────────────────────
    st.subheader("Worker Distance to Aerolift — Full Timeline")
    if len(prox_tl) > 2:
        ts = [p["t"] for p in prox_tl]
        ds = [p["d"] for p in prox_tl]
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=ts, y=ds, mode="lines", name="Min Distance (m)",
            line=dict(color="#00ccff", width=2),
            fill="tozeroy", fillcolor="rgba(0,200,255,0.06)"))
        fig_tl.add_hline(y=0.60, line_dash="dash", line_color="red",
                         annotation_text="Caution (0.6m)", annotation_font_color="red")
        fig_tl.add_hline(y=2.00, line_dash="dot",  line_color="orange",
                         annotation_text="Warning (2.0m)", annotation_font_color="orange")
        fig_tl.update_layout(
            xaxis_title="Time (s)", yaxis_title="Distance (m)",
            height=280, margin=dict(t=20, b=30, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color="#ccc"), showlegend=False)
        st.plotly_chart(fig_tl, use_container_width=True)
        prox_df = pd.DataFrame(prox_tl).rename(columns={"t":"Time (s)","d":"Distance (m)"})
        st.download_button("Proximity Timeline (CSV)", prox_df.to_csv(index=False),
                           "proximity_timeline.csv", "text/csv")
    else:
        st.info("No proximity data (no aerolift detected in video).")

    # ── Incident Heatmap — caution/warning per minute ────────────────────────
    st.subheader("Incident Heatmap — Events Per Minute")
    if cpm or warnpm:
        all_mins = sorted(set(list(cpm.keys()) + list(warnpm.keys())), key=lambda x: int(x))
        c_vals   = [cpm.get(m, 0)    for m in all_mins]
        w_vals   = [warnpm.get(m, 0) for m in all_mins]
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Bar(
            x=[f"{m}m" for m in all_mins],
            y=c_vals,
            name="Caution",
            marker_color="#cc0000",
            text=c_vals,
            textposition="outside"))
        fig_heat.add_trace(go.Bar(
            x=[f"{m}m" for m in all_mins],
            y=w_vals,
            name="Warning",
            marker_color="#ff9900",
            text=w_vals,
            textposition="outside"))
        peak_min  = max(all_mins, key=lambda m: cpm.get(m, 0) + warnpm.get(m, 0))
        peak_val  = cpm.get(peak_min, 0) + warnpm.get(peak_min, 0)
        fig_heat.update_layout(
            barmode="stack",
            xaxis_title="Minute", yaxis_title="Events",
            height=280, margin=dict(t=20, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color="#ccc"), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(f"Peak incident minute: **{peak_min}m** with {peak_val} total events "
                   f"({cpm.get(peak_min,0)} caution + {warnpm.get(peak_min,0)} warning)")
        heat_rows = [{"Minute": f"{m}m", "Caution Events": cpm.get(m,0),
                      "Warning Events": warnpm.get(m,0),
                      "Total": cpm.get(m,0)+warnpm.get(m,0)} for m in all_mins]
        heat_df = pd.DataFrame(heat_rows)
        st.download_button("Incident Heatmap (CSV)", heat_df.to_csv(index=False),
                           "incident_heatmap.csv", "text/csv")
    else:
        st.info("No incident data recorded by minute.")

    st.markdown("---")

    # ── Workers per minute ────────────────────────────────────────────────────
    st.subheader("Workers On-Site Per Minute")
    if wpm:
        mins_i = sorted(int(k) for k in wpm.keys())
        wvals  = [wpm.get(str(m), wpm.get(m, 0)) for m in mins_i]
        fig_wm = go.Figure(go.Bar(
            x=[f"{m}m" for m in mins_i], y=wvals,
            marker_color="#4488ff", text=wvals, textposition="outside"))
        fig_wm.update_layout(
            xaxis_title="Minute", yaxis_title="Max Workers",
            height=220, margin=dict(t=20, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color="#ccc"))
        st.plotly_chart(fig_wm, use_container_width=True)

    st.markdown("---")

    # ── Summary stats table ────────────────────────────────────────────────────
    st.subheader("Summary Statistics")
    rate_d = round(d / max(duration_min, 0.01), 2)
    rate_w = round(w / max(duration_min, 0.01), 2)
    summary_rows = [
        {"Metric": "Duration Analysed",     "Value": f"{duration_min} min"},
        {"Metric": "Camera / Resolution",   "Value": f"{cam_type} | {video_res}"},
        {"Metric": "Calibration (px/m)",    "Value": f"{ppm} px/m  (ScaleY={scale_y})"},
        {"Metric": "Caution Rate",          "Value": f"{rate_d} events/min"},
        {"Metric": "Warning Rate",          "Value": f"{rate_w} events/min"},
        {"Metric": "PPE Compliance",        "Value": f"{ppe_pct}%"},
        {"Metric": "Closest Ever (real min)",   "Value": f"{true_min_str} at {close_t}"},
        {"Metric": "Caution Zone Workers",  "Value": str(cum_c)},
        {"Metric": "Warning Zone Workers",  "Value": str(cum_w)},
        {"Metric": "Safe Zone Workers",     "Value": str(cum_s)},
        {"Metric": "Total Caution Events",  "Value": str(d)},
        {"Metric": "Total Warning Events",  "Value": str(w)},
        {"Metric": "Total No-PPE",          "Value": str(np_)},
        {"Metric": "Frames Processed",      "Value": f"{fc:,}"},
        {"Metric": "Site Safety Index",     "Value": f"{risk}/100 — {'LOW SAFETY INDEX' if risk<30 else 'MEDIUM SAFETY INDEX' if risk<70 else 'HIGH SAFETY INDEX'}"},
    ]
    sum_df = pd.DataFrame(summary_rows)
    st.dataframe(sum_df, use_container_width=True, hide_index=True)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("Summary Table (CSV)", sum_df.to_csv(index=False),
                           "safety_summary.csv", "text/csv", use_container_width=True)
    with col_dl2:
        summary_txt = "\n".join(f"{r['Metric']:30s}: {r['Value']}" for r in summary_rows)
        st.download_button("Summary (TXT)", summary_txt,
                           "safety_summary.txt", "text/plain", use_container_width=True)

    # ── Event log ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Event Log")
    if event_log:
        caution_evts = [e for e in event_log if "CAUTION" in e]
        warn_evts    = [e for e in event_log if "WARN" in e and "CAUTION" not in e]
        tab_all, tab_cau, tab_w = st.tabs([
            f"All ({len(event_log)})",
            f"Caution ({len(caution_evts)})",
            f"Warning ({len(warn_evts)})",
        ])
        with tab_all:
            st.text_area("All events", "\n".join(event_log[-300:]), height=220)
        with tab_cau:
            st.text_area("Caution events", "\n".join(caution_evts[-100:]) if caution_evts else "None", height=180)
        with tab_w:
            st.text_area("Warning events", "\n".join(warn_evts[-100:]) if warn_evts else "None", height=180)

        evt_rows = []
        for e in event_log:
            parts  = e.split("]", 1)
            ts_str = parts[0].replace("[","").strip() if len(parts)>1 else ""
            body   = parts[1].strip() if len(parts)>1 else e
            level  = "CAUTION" if "CAUTION" in body else "WARNING" if "WARN" in body else "INFO"
            evt_rows.append({"Timestamp":ts_str,"Level":level,"Detail":body})
        evt_df = pd.DataFrame(evt_rows)
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button("Event Log (CSV)", evt_df.to_csv(index=False),
                               "safety_events.csv", "text/csv", use_container_width=True)
        with col_e2:
            st.download_button("Event Log (TXT)", "\n".join(event_log),
                               "safety_events.txt", "text/plain", use_container_width=True)
    else:
        st.success("No proximity or PPE events detected.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🎬 Video Source")
    video_files = list(Path("video").glob("*.MP4")) + list(Path("video").glob("*.mp4"))
    video_options = [str(v) for v in video_files]
    if video_options:
        selected_video = st.selectbox("Select Video", video_options)
    else:
        selected_video = None
        st.info("No videos available yet — loading…", icon="⏳")

    # ── Device selection ──────────────────────────────────────────────────────
    def get_available_devices():
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append("mps")
        return devices

    available_devices = get_available_devices()
    default_device = (
        "mps" if "mps" in available_devices
        else next((d for d in available_devices if d.startswith("cuda")), "cpu")
    )
    selected_device = st.sidebar.selectbox(
        "Inference device",
        options=available_devices,
        index=available_devices.index(default_device),
        help="MPS uses Apple Silicon GPU, CUDA uses NVIDIA GPU, CPU is universal fallback.",
    )
    st.sidebar.caption(f"Running on: `{selected_device}`")

    # ── Progress section ──────────────────────────────────────────────────────
    st.divider()
    st.header("📊 Progress Settings")
    import config
    config.FRAME_SAMPLE_INTERVAL = st.slider(
        "Frame Sample Interval", 1, 120, config.FRAME_SAMPLE_INTERVAL,
        help="Process every Nth frame.",
    )
    config.DETECTION_CONFIDENCE_THRESHOLD = st.slider(
        "Detection Confidence", 0.1, 0.9, config.DETECTION_CONFIDENCE_THRESHOLD, 0.05,
    )
    update_every = st.slider("Dashboard Refresh (frames)", 5, 100, 30)
    run_progress = st.button(
        "▶ Run Progress Analysis", type="primary", use_container_width=True,
    )

    # ── Productivity section ──────────────────────────────────────────────────
    st.divider()
    st.header("⚡ Productivity Settings")
    prod_frame_skip = st.slider(
        "Frame Skip", 1, 30, 6,
        help="Process every Nth frame. Match to your timelapse cadence.",
    )
    prod_timelapse_interval = st.slider(
        "Timelapse Interval (s/frame)", 1, 15, 5,
        help="Real-world seconds each timelapse frame represents.",
    )
    prod_update_every = st.slider(
        "Productivity Refresh (frames)", 5, 100, 30,
        help="Rebuild live charts every N processed frames.",
    )
    run_productivity = st.button(
        "▶ Run Productivity Analysis", type="primary", use_container_width=True,
    )

    # ── Safety section ────────────────────────────────────────────────────────
    st.divider()
    st.header("🦺 Safety Settings")
    safety_frame_skip = st.slider(
        "Safety Frame Skip", 1, 30, 5,
        help="Process every Nth frame for safety detection.",
    )
    safety_update_every = st.slider(
        "Safety Alert Refresh (frames)", 5, 100, 20,
        help="Rebuild alert panel every N processed frames.",
    )
    run_safety = st.button(
        "▶ Run Safety Analysis", type="primary", use_container_width=True,
    )

    # ── Stage legend ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Stage Legend:**")
    for name, color in STAGE_COLORS.items():
        st.markdown(
            f'<span style="color:{color};font-weight:bold;">&#9632;</span> {name}',
            unsafe_allow_html=True,
        )


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🏗️ Precast Tunnel Segment — Construction Monitor")

tab_progress, tab_productivity, tab_safety = st.tabs([
    "📊 Progress Tracking",
    "⚡ Productivity Analytics",
    "🦺 Safety Monitoring",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROGRESS TRACKING
# ══════════════════════════════════════════════════════════════════════════════
with tab_progress:
    if run_progress and selected_video is None:
        st.warning("Please select a video from the sidebar first.", icon="📹")
        run_progress = False

    if run_progress:
        from video_processor import process_video

        st.markdown("---")
        col_video, col_status = st.columns([3, 2])
        with col_video:
            st.subheader("Live Detection Feed")
            frame_display = st.empty()
            progress_bar  = st.progress(0, text="Starting...")
        with col_status:
            st.subheader("Latest Mould Status")
            status_cards      = st.empty()
            date_display      = st.empty()
            timestamp_display = st.empty()
            stats_display     = st.empty()

        st.subheader("Segments Completed (Mould-wise)")
        segment_placeholder = st.empty()

        st.subheader("Timeline (building live...)")
        gantt_placeholder = st.empty()

        col_dur, col_comp = st.columns(2)
        with col_dur:
            duration_placeholder = st.empty()
        with col_comp:
            completion_placeholder = st.empty()

        col_table, col_log = st.columns(2)
        with col_table:
            st.subheader("Detailed Progress Report")
            table_placeholder = st.empty()
        with col_log:
            st.subheader("Recent Detections")
            log_placeholder = st.empty()

        recent_detections = []

        def frame_callback(info):
            frame_idx  = info["frame_idx"]
            total      = info["total_frames"]
            processed  = info["processed_count"]
            tracker    = info["tracker"]
            annotated  = info["annotated_frame"]
            timestamp  = info["timestamp"]
            date_str   = info.get("date")
            detections = info["detections"]
            ocr_fails  = info["ocr_failures"]

            pct = min(frame_idx / total, 1.0)
            progress_bar.progress(pct, text=f"Frame {frame_idx:,}/{total:,} | Processed: {processed:,}")

            for d in detections:
                recent_detections.append(
                    f"`{timestamp}` Mould {d['mould']+1}: **{d['class']}** ({d['confidence']:.0%})"
                )

            has_detections  = len(detections) > 0
            is_chart_update = (processed % update_every == 0) or (frame_idx >= total - 1)

            if has_detections or is_chart_update:
                disp     = cv2.resize(annotated, (720, 405))
                disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                frame_display.image(disp_rgb, channels="RGB", use_column_width=True)

                latest    = tracker.get_latest_stages()
                card_html = '<div style="display:flex;gap:12px;">'
                for m_idx in range(tracker.num_moulds):
                    stage = latest.get(m_idx, "Waiting...")
                    color = STAGE_COLORS.get(stage, "#888")
                    card_html += (
                        f'<div style="flex:1;padding:16px;border-radius:8px;'
                        f'border-left:5px solid {color};background:#f0f2f6;text-align:center;">'
                        f'<div style="font-size:14px;color:#666;">Mould {m_idx+1}</div>'
                        f'<div style="font-size:18px;font-weight:bold;color:{color};">{stage or "—"}</div>'
                        f'</div>'
                    )
                card_html += "</div>"
                status_cards.markdown(card_html, unsafe_allow_html=True)

                if date_str:
                    date_display.markdown(f"**Date:** {date_str}")
                timestamp_display.markdown(f"**Video Time:** `{timestamp}`")
                stats_display.markdown(
                    f"OCR failures: {ocr_fails} | "
                    f"Detections logged: {sum(len(v) for v in tracker.raw_observations.values())}"
                )
                if recent_detections:
                    log_placeholder.markdown("\n\n".join(recent_detections[-15:]))

            if is_chart_update:
                interim = get_interim_timelines(tracker)

                seg_counts, _ = compute_segment_counts(interim)
                seg_html = '<div style="display:flex;gap:12px;margin-top:8px;">'
                for m_name, cnt in seg_counts.items():
                    seg_html += (
                        f'<div style="flex:1;padding:10px;border-radius:8px;'
                        f'background:#e8f5e9;text-align:center;">'
                        f'<div style="font-size:12px;color:#666;">{m_name}</div>'
                        f'<div style="font-size:20px;font-weight:bold;color:#2e7d32;">{cnt}</div>'
                        f'<div style="font-size:11px;color:#888;">segment{"s" if cnt != 1 else ""}</div>'
                        f'</div>'
                    )
                seg_html += "</div>"
                segment_placeholder.markdown(seg_html, unsafe_allow_html=True)

                fig, df = build_gantt_chart(interim)
                if fig:
                    gantt_placeholder.plotly_chart(fig, use_container_width=True, key=f"gantt_{processed}")
                if df is not None and not df.empty:
                    dur_fig = build_stage_duration_chart(df)
                    if dur_fig:
                        duration_placeholder.plotly_chart(dur_fig, use_container_width=True, key=f"dur_{processed}")
                comp_fig = build_completion_matrix(interim)
                if comp_fig:
                    completion_placeholder.plotly_chart(comp_fig, use_container_width=True, key=f"comp_{processed}")
                detail_df = build_detailed_table(interim)
                if not detail_df.empty:
                    table_placeholder.dataframe(detail_df, use_container_width=True, hide_index=True)

        result = process_video(selected_video, frame_callback=frame_callback)
        progress_bar.progress(1.0, text="Processing complete!")

        st.markdown("---")
        st.header("Final Report")
        render_report_body(result["timelines"], result["metadata"])

    else:
        data = load_results()
        if data is None:
            st.info("No results found. Select a video and click **▶ Run Progress Analysis** to start.", icon="📹")
        else:
            render_report_body(data["timelines"], data["metadata"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRODUCTIVITY ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_productivity:
    # Import only what the live callback needs; charts/render use the helper function
    from productivity_processor import (
        process_productivity,
        build_lean_bar_chart,
        build_crew_balance_chart,
        PHASE_COLORS as PROD_PHASE_COLORS,
    )

    # ── Session state init ────────────────────────────────────────────────────
    if "prod_boundary_pts" not in st.session_state:
        st.session_state.prod_boundary_pts = []
    if "prod_canvas_scale" not in st.session_state:
        st.session_state.prod_canvas_scale = (1.0, 1.0)

    # ── Helper: read first frame from selected video ───────────────────────────
    def _get_first_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (frame, w, h) if ret else (None, 0, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # LIVE PROCESSING MODE
    # ─────────────────────────────────────────────────────────────────────────
    if run_productivity and selected_video is None:
        st.warning("Please select a video from the sidebar first.", icon="📹")
        run_productivity = False

    if run_productivity:
        boundary_pts = st.session_state.prod_boundary_pts

        if len(boundary_pts) < 2:
            st.error(
                "⚠️ Boundary not set (fewer than 2 points). "
                "Please draw a boundary on the canvas below, then run analysis again."
            )
            # Fall through to show the canvas so the user can draw
        else:
            st.markdown("---")

            # Layout mirrors the progress tab
            col_vid, col_stat = st.columns([3, 2])
            with col_vid:
                st.subheader("Live Detection Feed")
                prod_frame_display = st.empty()
                prod_progress_bar  = st.progress(0, text="Loading models...")

            with col_stat:
                st.subheader("Mould Status")
                prod_status_cards  = st.empty()
                st.subheader("System Diagnostics")
                prod_diagnostics   = st.empty()

            # Running lean ratio (live)
            st.subheader("Running Lean Ratio")
            prod_lean_bar_ph = st.empty()

            # Crew balance timeline (live)
            st.subheader("Crew Balance Timeline (building live...)")
            prod_crew_ph = st.empty()

            def prod_frame_callback(info):
                frame_idx  = info["frame_idx"]
                total      = info["total_frames"]
                annotated  = info["annotated_frame"]
                phases     = info["confirmed_phases"]
                zone_cnt   = info["workers_in_zone"]
                active     = info["active_workers"]
                ghost      = info["ghost_count"]
                blacklisted = info["blacklist_count"]
                timeline   = info["timeline_history"]
                processed  = sum(len(v) for v in timeline.values())

                pct = min(frame_idx / max(total, 1), 1.0)
                prod_progress_bar.progress(
                    pct, text=f"Frame {frame_idx:,}/{total:,}"
                )

                # Always update the live feed
                disp     = cv2.resize(annotated, (720, 405))
                disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                prod_frame_display.image(disp_rgb, channels="RGB", use_column_width=True)

                # Mould status cards (phase + worker count)
                card_html = '<div style="display:flex;flex-direction:column;gap:10px;">'
                for i, (mould_id, phase) in enumerate(phases.items()):
                    workers = zone_cnt.get(mould_id, 0)
                    color   = PROD_PHASE_COLORS.get(phase, "#888")
                    card_html += (
                        f'<div style="padding:12px;border-radius:8px;'
                        f'border-left:5px solid {color};background:#f0f2f6;">'
                        f'<div style="font-size:13px;color:#555;">Mould {mould_id}</div>'
                        f'<div style="font-size:16px;font-weight:bold;color:{color};">{phase}</div>'
                        f'<div style="font-size:12px;color:#777;">👷 {workers} worker{"s" if workers != 1 else ""} in zone</div>'
                        f'</div>'
                    )
                card_html += "</div>"
                prod_status_cards.markdown(card_html, unsafe_allow_html=True)

                # System diagnostics
                prod_diagnostics.markdown(
                    f"**Active workers:** {active} &nbsp;|&nbsp; "
                    f"**Ghost memory:** {ghost} &nbsp;|&nbsp; "
                    f"**Blacklisted:** {blacklisted}"
                )

                # Throttled chart updates
                is_chart_update = (frame_idx % (prod_update_every * prod_frame_skip) == 0) or (frame_idx >= total - 1)
                if is_chart_update and timeline:
                    lean_fig = build_lean_bar_chart(timeline)
                    if lean_fig:
                        prod_lean_bar_ph.plotly_chart(
                            lean_fig, use_container_width=True, key=f"lean_{frame_idx}"
                        )
                    crew_fig = build_crew_balance_chart(
                        timeline, prod_frame_skip, prod_timelapse_interval
                    )
                    if crew_fig:
                        prod_crew_ph.plotly_chart(
                            crew_fig, use_container_width=True, key=f"crew_{frame_idx}"
                        )

            # Run the productivity pipeline
            result = process_productivity(
                selected_video,
                boundary_points=boundary_pts,
                frame_skip=prod_frame_skip,
                timelapse_interval=prod_timelapse_interval,
                frame_callback=prod_frame_callback,
                device=selected_device,
            )
            prod_progress_bar.progress(1.0, text="Analysis complete!")

            # Save to disk so results survive page navigation (same pattern as progress tab)
            save_productivity_results(result)

            st.markdown("---")
            st.header("Final Productivity Report")
            render_productivity_report_body(load_productivity_results())

    # ─────────────────────────────────────────────────────────────────────────
    # BOUNDARY DRAWING / IDLE STATE
    # ─────────────────────────────────────────────────────────────────────────
    if not run_productivity or len(st.session_state.prod_boundary_pts) < 2:
        if run_productivity:
            st.markdown("---")  # separator after the error message above

        st.subheader("Step 1 — Draw the Active Zone Boundary")
        st.markdown(
            "Click on the image below to place boundary points **from left to right**. "
            "Workers whose feet are **above** the line will be ignored. "
            "Place at least 2 points, then click **▶ Run Productivity Analysis** in the sidebar."
        )

        first_frame_prod, fw, fh = _get_first_frame(selected_video)

        if first_frame_prod is None:
            st.error("Cannot read the selected video. Check the path in the sidebar.")
        else:
            # Scale to a fixed canvas width
            CANVAS_W = 720
            CANVAS_H = int(fh * CANVAS_W / fw)
            scale_x  = fw / CANVAS_W
            scale_y  = fh / CANVAS_H
            st.session_state.prod_canvas_scale = (scale_x, scale_y)

            frame_rgb   = cv2.cvtColor(first_frame_prod, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (CANVAS_W, CANVAS_H))
            pil_img     = Image.fromarray(frame_small)

            try:
                from streamlit_drawable_canvas import st_canvas

                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 200, 0.4)",
                    stroke_width=2,
                    stroke_color="#00ffcc",
                    background_image=pil_img,
                    drawing_mode="point",
                    point_display_radius=6,
                    height=CANVAS_H,
                    width=CANVAS_W,
                    key="boundary_canvas",
                )

                # Extract and persist boundary points
                pts_canvas = []
                if canvas_result.json_data:
                    for obj in canvas_result.json_data.get("objects", []):
                        if obj.get("type") == "circle":
                            r  = obj.get("radius", 6)
                            px = obj.get("left", 0) + r
                            py = obj.get("top",  0) + r
                            pts_canvas.append((px, py))
                pts_canvas.sort(key=lambda p: p[0])

                pts_full = [
                    (int(px * scale_x), int(py * scale_y))
                    for px, py in pts_canvas
                ]
                st.session_state.prod_boundary_pts = pts_full

                # Visual feedback
                col_info, col_clear = st.columns([4, 1])
                with col_info:
                    if len(pts_full) >= 2:
                        st.success(
                            f"✅ Boundary set — {len(pts_full)} points. "
                            "Click **▶ Run Productivity Analysis** in the sidebar."
                        )
                    elif len(pts_full) == 1:
                        st.warning("Add at least 1 more point.")
                    else:
                        st.info("Click on the image to place boundary points.")
                with col_clear:
                    if st.button("🗑 Clear", help="Reset the boundary"):
                        st.session_state.prod_boundary_pts = []
                        st.rerun()

                # Preview boundary on frame (show in an expander)
                if len(pts_full) >= 2:
                    with st.expander("Preview boundary on full frame", expanded=False):
                        preview = frame_rgb.copy()
                        pts_sorted = sorted(pts_canvas, key=lambda p: p[0])
                        for i in range(1, len(pts_sorted)):
                            p1 = (int(pts_sorted[i-1][0]), int(pts_sorted[i-1][1]))
                            p2 = (int(pts_sorted[i][0]),   int(pts_sorted[i][1]))
                            cv2.line(preview, p1, p2, (0, 255, 200), 2)
                        for pt in pts_sorted:
                            cv2.circle(preview, (int(pt[0]), int(pt[1])), 5, (0, 200, 180), -1)
                        st.image(preview, use_column_width=True)

            except ImportError:
                st.error(
                    "**streamlit-drawable-canvas** is not installed. "
                    "Run: `pip install streamlit-drawable-canvas` and restart the app."
                )
                st.image(frame_small, caption="Select video frame (install canvas library to draw boundary)")

        # ── Load saved results from a previous run (independent of whether
        #    processing is currently active — mirrors the progress tab's else branch)
        prod_saved = load_productivity_results()
        if prod_saved:
            st.divider()
            st.header("Last Analysis Results")
            render_productivity_report_body(prod_saved)
        else:
            st.info(
                "No previous analysis found. Draw the boundary above and click "
                "**▶ Run Productivity Analysis** to generate results.",
                icon="⚡",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SAFETY MONITORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_safety:
    from safety_processor import process_safety

    # ─────────────────────────────────────────────────────────────────────────
    # LIVE PROCESSING MODE
    # ─────────────────────────────────────────────────────────────────────────
    if run_safety and selected_video is None:
        st.warning("Please select a video from the sidebar first.", icon="📹")
        run_safety = False

    if run_safety:
        st.markdown("---")

        col_vid, col_alerts = st.columns([3, 2])
        with col_vid:
            st.subheader("Live Safety Feed")
            safety_frame_disp = st.empty()
            safety_prog_bar   = st.progress(0, text="Loading models...")

        with col_alerts:
            st.subheader("Real-Time Alerts")
            safety_alert_box  = st.empty()
            st.subheader("Cumulative Counters")
            safety_counters   = st.empty()

        st.subheader("Recent Event Log (live)")
        safety_log_ph = st.empty()

        # Extra live chart placeholders
        safety_timeline_ph = st.empty()
        safety_metrics_ph  = st.empty()

        recent_safety_events = []

        PROX_CHK = 4.0  # must match safety_processor.py constant

        def safety_frame_callback(info):
            frame_idx  = info["frame_idx"]
            total      = info["total_frames"]
            annotated  = info["annotated_frame"]
            alerts     = info["frame_alerts"]
            cum_d      = info["cumul_danger"]
            cum_w      = info["cumul_warning"]
            cum_nppe   = info["cumul_no_ppe"]
            total_wkrs = info["total_workers"]
            ppe_pct    = info.get("ppe_compliance_pct", 0)
            closest    = info.get("closest_approach", None)
            close_t    = info.get("closest_time", "—")
            prox_tl    = info.get("proximity_timeline", [])
            wpm        = info.get("workers_per_minute", {})
            n_cau      = info.get("n_caution", 0)
            n_warn_z   = info.get("n_warning", 0)
            n_safe          = info.get("n_safe", 0)
            n_near_aero     = info.get("n_near_aerolift", 0)
            n_frame_ok      = info.get("frame_ppe_ok", 0)
            n_frame_fail    = info.get("frame_ppe_fail", 0)
            cam_type        = info.get("cam_type", "—")
            ppm        = info.get("ppm", 0)

            pct = min(frame_idx / max(total, 1), 1.0)
            safety_prog_bar.progress(pct, text=f"Frame {frame_idx:,}/{total:,}")

            # Live feed
            disp_rgb = cv2.cvtColor(
                cv2.resize(annotated, (720, 405)), cv2.COLOR_BGR2RGB)
            safety_frame_disp.image(disp_rgb, channels="RGB", use_column_width=True)

            for a in alerts:
                recent_safety_events.append(a)

            # Alert panel
            if recent_safety_events:
                alert_html = '<div style="display:flex;flex-direction:column;gap:5px;">'
                for evt in recent_safety_events[-8:][::-1]:
                    if "CAUTION" in evt:  bg, border = "#cb9292", "#ff4444"
                    elif "WARN" in evt:   bg, border = "#b9a676", "#ff9900"
                    else:                 bg, border = "#9cd1aa", "#33cc66"
                    alert_html += (f'<div style="padding:7px;border-radius:5px;'
                                   f'border-left:4px solid {border};background:{bg};'
                                   f'font-size:11px;font-family:monospace;">{evt}</div>')
                alert_html += "</div>"
                safety_alert_box.markdown(alert_html, unsafe_allow_html=True)
            else:
                safety_alert_box.info("No alerts yet.")

            is_chart_update = (
                frame_idx % (safety_update_every * safety_frame_skip) == 0
                or frame_idx >= total - 1)

            if is_chart_update:
                rob_cl      = info.get("robust_closest", None)
                closest_str = f"{rob_cl*100:.0f} cm" if rob_cl and rob_cl > 0.05 else (
                              f"{closest*100:.0f} cm" if closest and closest > 0.05 else "—")
                bar_col = "#22cc55" if ppe_pct>=80 else "#ff9900" if ppe_pct>=50 else "#cc0000"

                near_col = "#cc0000" if n_cau > 0 else "#ff9900" if n_near_aero > 0 else "#22cc55"

                cnt_html = (
                    f'<div style="padding:10px;border-radius:8px;border:2px solid #888;'
                    f'background:#1a1a1a;text-align:center;margin-bottom:8px;">'
                    f'<div style="font-size:10px;color:#aaa;letter-spacing:1px;">WORKERS NEAR AEROLIFT</div>'
                    f'<div style="font-size:30px;font-weight:bold;color:#ffffff;">{n_near_aero}</div>'
                    f'<div style="font-size:10px;color:#888;">within {PROX_CHK}m — this frame</div></div>'

                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:8px;">'
                    f'<div style="padding:7px;background:#1a0000;border-radius:5px;text-align:center;border-top:2px solid #cc0000;">'
                    f'<div style="font-size:9px;color:#cc8888;">CAUTION</div>'
                    f'<div style="font-size:20px;font-weight:bold;color:#cc0000;">{cum_d}</div>'
                    f'<div style="font-size:9px;color:#886666;">cumulative total</div></div>'
                    f'<div style="padding:7px;background:#1a1000;border-radius:5px;text-align:center;border-top:2px solid #ff9900;">'
                    f'<div style="font-size:9px;color:#cc9944;">WARNING</div>'
                    f'<div style="font-size:20px;font-weight:bold;color:#ff9900;">{cum_w}</div>'
                    f'<div style="font-size:9px;color:#887744;">cumulative total</div></div>'
                    f'<div style="padding:7px;background:#001a00;border-radius:5px;text-align:center;border-top:2px solid #22cc55;">'
                    f'<div style="font-size:9px;color:#448844;">PPE OK</div>'
                    f'<div style="font-size:20px;font-weight:bold;color:#22cc55;">{n_frame_ok}</div>'
                    f'<div style="font-size:9px;color:#337733;">wearing PPE · frame</div></div>'
                    f'</div>'

                    f'<div style="padding:7px;background:#fff0f0;border-radius:5px;text-align:center;margin-bottom:8px;">'
                    f'<div style="font-size:9px;color:#888;">No PPE (cumulative total)</div>'
                    f'<div style="font-size:22px;font-weight:bold;color:#cc0000;">{cum_nppe}</div></div>'

                    f'<div style="padding:8px;background:#1a1a2e;border-radius:5px;margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:10px;color:#aaa;">PPE Compliance</span>'
                    f'<span style="font-size:13px;font-weight:bold;color:{bar_col};">{ppe_pct}%</span></div>'
                    f'<div style="background:#333;border-radius:3px;height:10px;margin-top:4px;">'
                    f'<div style="background:{bar_col};width:{min(100,int(ppe_pct))}%;height:10px;border-radius:3px;"></div></div></div>'
                )
                safety_counters.markdown(cnt_html, unsafe_allow_html=True)

                # Live proximity timeline chart
                if len(prox_tl) > 3:
                    ts = [p["t"] for p in prox_tl]
                    ds = [p["d"] for p in prox_tl]
                    fig_tl = go.Figure()
                    fig_tl.add_trace(go.Scatter(
                        x=ts, y=ds, mode="lines",
                        line=dict(color="#00ccff", width=2),
                        fill="tozeroy", fillcolor="rgba(0,200,255,0.06)"))
                    fig_tl.add_hline(y=0.60, line_dash="dash", line_color="red")
                    fig_tl.add_hline(y=2.00, line_dash="dot",  line_color="orange")
                    fig_tl.update_layout(
                        title="Distance to Aerolift (m)",
                        height=200, margin=dict(t=30,b=20,l=40,r=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.05)",
                        font=dict(color="#ccc",size=10), showlegend=False,
                        xaxis_title="Time (s)", yaxis_title="m")
                    safety_timeline_ph.plotly_chart(fig_tl, use_container_width=True)

            if is_chart_update:
                # Live zone count chart
                zone_tl_live = info.get("zone_count_timeline", [])
                if len(zone_tl_live) > 2:
                    _zt  = [p["t"]       for p in zone_tl_live]
                    _zc  = [p["caution"] for p in zone_tl_live]
                    _zw  = [p["warning"] for p in zone_tl_live]
                    fig_zt = go.Figure()
                    fig_zt.add_trace(go.Scatter(
                        x=_zt, y=_zc, mode="lines", name="Caution",
                        line=dict(color="#cc0000", width=2),
                        fill="tozeroy", fillcolor="rgba(204,0,0,0.1)"))
                    fig_zt.add_trace(go.Scatter(
                        x=_zt, y=_zw, mode="lines", name="Warning",
                        line=dict(color="#ff9900", width=2),
                        fill="tozeroy", fillcolor="rgba(255,153,0,0.07)"))
                    fig_zt.update_layout(
                        title="Workers in Caution/Warning Zone (live)",
                        height=200, margin=dict(t=30, b=20, l=40, r=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.05)",
                        font=dict(color="#ccc", size=10),
                        legend=dict(orientation="h", y=1.15),
                        xaxis_title="Time (s)", yaxis_title="Workers",
                        yaxis=dict(rangemode="nonnegative"))
                    safety_metrics_ph.plotly_chart(fig_zt, use_container_width=True)

                # Aerolift active time live counter
                _al_sec = info.get("aerolift_active_sec", 0)
                if _al_sec > 0:
                    safety_timeline_ph.markdown(
                        f'<div style="padding:8px;border-radius:6px;background:#1a1a2e;'
                        f'border-left:3px solid #88ccff;margin-top:6px;">'
                        f'<span style="font-size:10px;color:#aaa;">Aerolift Active Time</span><br/>'
                        f'<span style="font-size:18px;font-weight:bold;color:#88ccff;">{_al_sec}s</span>'
                        f'<span style="font-size:11px;color:#666;"> ({round(_al_sec/60,1)} min)</span>'
                        f'</div>',
                        unsafe_allow_html=True)

            if recent_safety_events:
                safety_log_ph.text_area(
                    "Events", "\n".join(recent_safety_events[-50:]),
                    height=180, key=f"slog_{frame_idx}")

        result = process_safety(
            selected_video,
            frame_skip=safety_frame_skip,
            frame_callback=safety_frame_callback,
        )
        safety_prog_bar.progress(1.0, text="Safety analysis complete!")
        save_safety_results(result)

        st.markdown("---")
        st.header("Final Safety Report")
        render_safety_report_body(load_safety_results())

    # ─────────────────────────────────────────────────────────────────────────
    # IDLE STATE — show saved results (if any)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        safety_saved = load_safety_results()
        if safety_saved:
            st.subheader("Last Safety Analysis Results")
            render_safety_report_body(safety_saved)
        else:
            st.info(
                "No previous safety analysis found. "
                "Select a video and click **▶ Run Safety Analysis** in the sidebar.",
                icon="🦺",
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Danger Events",  "—", help="Run analysis to populate")
            c2.metric("PPE Violations", "—", help="Run analysis to populate")
            c3.metric("Warning Events", "—", help="Run analysis to populate")
