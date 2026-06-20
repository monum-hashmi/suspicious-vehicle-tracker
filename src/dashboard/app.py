import sys
sys.path.insert(0, ".")

import os
import streamlit as st
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
from src.dashboard.db import init_db, get_all_alerts, clear_alerts

st.set_page_config(page_title="Suspicious Vehicle Tracker", layout="wide", page_icon="🛡️")
init_db()

with st.sidebar:
    st.markdown("## 🛡️ Sentinel")
    st.markdown("Suspicious vehicle & person tracker")
    st.divider()
    demo_mode = st.toggle("Demo data", value=False)
    if st.button("Clear Alerts"):
      clear_alerts()
      st.success("Alerts cleared")
    st.divider()
    st.markdown("**Team**")
    st.markdown("Monum — Detection & datasets")
    st.markdown("Areeba — Tracking & rule engine")
    st.markdown("Hafsa — Dashboard & reporting")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("YOLOv8 → DeepSORT → Rule engine")
    st.divider()
    st.markdown("**Thresholds**")
    st.markdown("Detection conf.: 0.25")
    st.markdown("Dwell: 45s")
    st.markdown("Motion: 0.3× vehicle width")

st.title("🛡️ Suspicious vehicle & person tracker")
st.caption("Real-time multi-class detection and behavioral alert system for flagged zones")

severities = ["Critical", "High", "Medium", "Unknown"]
severity_colors = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Unknown": "⚪"}


st.divider()

metrics_placeholder = st.container()
st.divider()
left, right = st.columns([2, 1])

with left:
    st.subheader("Video feed")
    clip_source = st.radio("Source", ["Upload a clip", "Choose from library"], horizontal=True)

    uploaded_file = None
    selected_clip_path = None

    if clip_source == "Upload a clip":
        uploaded_file = st.file_uploader("Upload a clip", type=["mp4"], label_visibility="collapsed")
    else:
        clips_dir = "clips"
        if os.path.exists(clips_dir):
            available_clips = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
            if available_clips:
                selected_clip_name = st.selectbox("Select a clip", available_clips)
                selected_clip_path = os.path.join(clips_dir, selected_clip_name)
            else:
                st.warning("No clips found in clips/ folder.")
        else:
            st.warning("clips/ folder not found.")

has_video = (uploaded_file is not None) or (selected_clip_path is not None)

current_video_name = uploaded_file.name if uploaded_file else (os.path.basename(selected_clip_path) if selected_clip_path else None)

if demo_mode:
    alerts = get_all_alerts()
elif has_video and current_video_name:
    all_db_alerts = get_all_alerts()
    alerts = [a for a in all_db_alerts if a[6] == current_video_name]
else:
    alerts = []

with metrics_placeholder:
    col1, col2, col3, col4, col5 = st.columns(5)
    person_ids = set(a[2] for a in alerts) if alerts else set()
    vehicle_ids = set(a[3] for a in alerts) if alerts else set()

    with col1:
        st.metric("Persons in alerts", len(person_ids))
    with col2:
        st.metric("Vehicles in alerts", len(vehicle_ids))
    with col3:
        st.metric("Suspicious events", len(alerts))
    with col4:
        st.metric("Processing FPS", "~7" if alerts else "0")
    with col5:
        st.metric("Detection threshold", "0.25")

with left:
    video_path = None

    if uploaded_file:
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.video(uploaded_file)
        with open("temp_upload.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_path = "temp_upload.mp4"
    elif selected_clip_path:
        st.session_state["uploaded_file_name"] = selected_clip_path
        st.video(selected_clip_path)
        video_path = selected_clip_path

    if video_path:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if ret:
            st.markdown("**Event timeline**")
            if alerts:
                timeline_df = pd.DataFrame(alerts, columns=["id", "timestamp", "person_id", "vehicle_id", "roi_id", "snapshot_path", "video_source", "rule_triggered", "dwell_time_at_trigger"])
                duration = frame_count / fps
                cols = st.columns(len(timeline_df))
                for i, (_, row) in enumerate(timeline_df.iterrows()):
                    with cols[i]:
                        if st.button(f"⚠ {round(row['timestamp'], 2)}s", key=f"marker_{row['id']}", use_container_width=True):
                            st.session_state["selected_alert"] = row["id"]
                st.caption(f"Clip duration: {duration:.1f}s — click a marker to jump to that event")
            else:
                st.caption("No events on timeline yet")

            st.subheader("Define ROI")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            col_x1, col_y1, col_x2, col_y2 = st.columns(4)
            with col_x1:
                x1 = st.number_input("X1", 0, w, int(w * 0.2))
            with col_y1:
                y1 = st.number_input("Y1", 0, h, int(h * 0.2))
            with col_x2:
                x2 = st.number_input("X2", 0, w, int(w * 0.6))
            with col_y2:
                y2 = st.number_input("Y2", 0, h, int(h * 0.6))

            overlay = frame_rgb.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
            st.image(overlay, caption="ROI preview")

            if st.button("Save ROI", type="primary"):
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                with open("roi.json", "w") as f:
                    json.dump({"polygon": polygon}, f)
                st.success("ROI saved to roi.json")

    if not video_path:
        st.container(border=True).markdown(
            "<div style='text-align:center; padding: 2rem 0;'>"
            "<div style='font-size: 40px;'>🎥</div>"
            "<p style='margin-top: 0.5rem;'>No analysis has been run yet.</p>"
            "<p style='font-size: 13px; opacity: 0.7;'>Upload a clip or select one from the library to begin.</p>"
            "</div>",
            unsafe_allow_html=True
        )

with right:
    st.subheader("Live alerts")

    if not alerts:
        st.container(border=True).markdown(
            "<div style='text-align:center; padding: 1.5rem 0;'>"
            "<div style='font-size: 32px;'>🔔</div>"
            "<p style='margin-top: 0.5rem;'>No alerts available.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        search = st.text_input("Search by person/vehicle ID", "")
        severity_filter = st.multiselect("Severity", severities, default=severities)

        for i, alert in enumerate(alerts):
            alert_id, ts, person_id, vehicle_id, roi_id, snapshot, source, rule_triggered, dwell_time_at_trigger = alert

            if dwell_time_at_trigger is None:
                severity = "Unknown"
            elif dwell_time_at_trigger >= 60:
                severity = "Critical"
            elif dwell_time_at_trigger >= 50:
                severity = "High"
            else:
                severity = "Medium"

            rule = rule_triggered or "Not recorded"

            if search and search not in str(person_id) and search not in str(vehicle_id):
                continue
            if severity not in severity_filter:
                continue

            with st.container(border=True):
                st.markdown(f"{severity_colors[severity]} **{severity}** — {round(ts, 2)}s")
                st.caption(f"Person {person_id} · Vehicle {vehicle_id} · ROI {roi_id}")
                st.caption(f"Rule: {rule}")
                if st.button("Investigate", key=f"inv_{i}", use_container_width=True):
                    st.session_state["selected_alert"] = alert_id

if "selected_alert" in st.session_state and alerts:
    st.divider()
    st.subheader("Alert investigation")

    df = pd.DataFrame(alerts, columns=["id", "timestamp", "person_id", "vehicle_id", "roi_id", "snapshot_path", "video_source", "rule_triggered", "dwell_time_at_trigger"])
    sel = df[df["id"] == st.session_state["selected_alert"]]

    if not sel.empty:
        row = sel.iloc[0]
        dwell = row["dwell_time_at_trigger"]

        if pd.isna(dwell):
            severity = "Unknown"
        elif dwell >= 60:
            severity = "Critical"
        elif dwell >= 50:
            severity = "High"
        else:
            severity = "Medium"

        rule = row["rule_triggered"] if pd.notna(row["rule_triggered"]) else "Not recorded"

        with st.container(border=True):
            col_img, col_meta = st.columns([1, 1])

            with col_img:
                st.markdown("**Frame snapshot**")
                if os.path.exists(row['snapshot_path']):
                    st.image(row['snapshot_path'], caption=f"Real crop: {row['snapshot_path']}")
                else:
                    placeholder = np.full((300, 400, 3), 40, dtype=np.uint8)
                    cv2.rectangle(placeholder, (120, 80), (280, 220), (0, 0, 255), 2)
                    cv2.putText(placeholder, "Person", (120, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.rectangle(placeholder, (60, 150), (180, 260), (0, 200, 0), 2)
                    cv2.putText(placeholder, "Vehicle", (60, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                    st.image(placeholder, caption=f"{row['snapshot_path']} (placeholder — real crop unavailable)")

            with col_meta:
                st.markdown(f"{severity_colors[severity]} **Severity:** {severity}")
                st.markdown(f"**Timestamp:** {round(row['timestamp'], 2)}s")
                st.markdown(f"**Person ID:** {row['person_id']}")
                st.markdown(f"**Vehicle ID:** {row['vehicle_id']}")
                st.markdown(f"**ROI:** {row['roi_id']}")
                dwell_display = f"{row['dwell_time_at_trigger']:.2f}s" if pd.notna(row['dwell_time_at_trigger']) else "Not recorded"
                st.markdown(f"**Dwell time:** {dwell_display}")
                st.markdown(f"**Rule triggered:** {rule}")
                st.markdown(f"**Confidence:** ≥ 0.25")
                st.markdown(f"**Source:** {row['video_source']}")
                st.download_button(
                    "Download evidence (JSON)",
                    json.dumps(row.to_dict(), indent=2),
                    file_name=f"alert_{row['id']}.json",
                    use_container_width=True
                )