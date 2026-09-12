"""Identification view: Biometric workstation, dual-mode optical acquisition (Webcam & Upload), decision panels, and rejection trace."""

import time
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.config import config
from app.recognition.recognizer import FaceRecognitionService, IdentificationOutput
from app.ui.components import (
    render_analysis_trace,
    render_camera_permission_notice,
    render_decision_panel,
    render_empty_state,
    render_face_alignment_guide,
    render_hardware_header,
    render_similarity_gauge,
    render_webcam_monitor_bezel,
    render_workflow_mode_badge,
)
from app.ui.theme import ICONS, THEME, render_html
from app.utils.image_utils import bgr_to_rgb, crop_face, load_image


def render_identify_view(rec_service: FaceRecognitionService):
    """Render the physical identification workstation with Live Webcam and File Upload acquisition modes."""
    curr_thresh = float(st.session_state.get("threshold", config.match_threshold))

    # Instrument Top Header Strip
    render_hardware_header(
        page_title="IDENTIFY TERMINAL",
        subtitle="OPTICAL SIGNAL ACQUISITION, EMBEDDING MATCHING, & UNKNOWN REJECTION AUDIT",
        model_status="READY",
        device=rec_service.device_info,
        threshold=curr_thresh,
    )

    # Workflow Mode Badge: Transient Query Mode Guarantee
    render_workflow_mode_badge("identification")

    # Operating Threshold Slider
    thresh_col1, thresh_col2 = st.columns([3, 2], vertical_alignment="center")
    with thresh_col1:
        st.markdown(
            f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; padding-top: 0.25rem;">
                <span style="color: {THEME['accent_amber']}; font-weight: 700;">OPTICAL INPUT CHANNELS:</span>
                Select Live Webcam scanner (Real-time / Snapshot) or Local Image Upload.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with thresh_col2:
        active_thresh = st.slider(
            "Operating Threshold (τ)",
            min_value=0.10,
            max_value=0.90,
            value=curr_thresh,
            step=0.01,
            key="identify_thresh_slider",
            help="Faces with cosine similarity >= τ are accepted as MATCH. Scores below τ are rejected as UNKNOWN.",
        )
        if active_thresh != curr_thresh:
            st.session_state["threshold"] = active_thresh
            curr_thresh = active_thresh

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # PRIMARY OPTICAL CHANNEL TABS (Webcam First for Instant Biometric Audit)
    # =========================================================================
    tab_webcam, tab_upload = st.tabs([
        "📷 [01] LIVE WEBCAM SCANNER",
        "📁 [02] IMAGE FILE UPLOAD",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: WEBCAM LIVE SCANNER (Real-Time Video Stream + Snapshot Capture)
    # -------------------------------------------------------------------------
    with tab_webcam:
        _render_webcam_identification_tab(rec_service, curr_thresh)

    # -------------------------------------------------------------------------
    # TAB 2: IMAGE FILE UPLOAD
    # -------------------------------------------------------------------------
    with tab_upload:
        _render_upload_identification_tab(rec_service, curr_thresh)


def _render_webcam_identification_tab(
    rec_service: FaceRecognitionService,
    curr_thresh: float,
):
    """Render the dual-engine webcam identification workstation."""
    # Privacy & Permission Guidance
    render_camera_permission_notice(is_denied=False)

    # Viewfinder Bezel & Alignment Reticle
    render_webcam_monitor_bezel(
        channel_name="CH-01 // WEBCAM LIVE VIEWPORT",
        status_text="ONLINE",
        led_color="green",
        device_label="INTEGRATED OPTICAL SENSOR",
    )
    render_face_alignment_guide()

    # Sub-Mode Selector: Continuous Live Feed vs High-Res Snapshot
    cam_mode = st.radio(
        "WEBCAM ACQUISITION ENGINE",
        ["⚡ REAL-TIME VIDEO STREAM (CONTINUOUS INFERENCE)", "📸 HIGH-RES SNAPSHOT CAPTURE"],
        horizontal=True,
        key="webcam_engine_selector",
    )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # SUB-MODE A: REAL-TIME CONTINUOUS VIDEO STREAM
    # ---------------------------------------------------------------------
    if "REAL-TIME" in cam_mode:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1], vertical_alignment="center")
        with ctrl_col1:
            run_live = st.toggle(
                "⚡ ENGAGE LIVE REAL-TIME OPTICAL SCANNER",
                value=False,
                key="live_cam_toggle",
                help="Start real-time continuous facial recognition directly on your connected webcam.",
            )
        with ctrl_col2:
            cam_idx = st.number_input("Sensor Index", min_value=0, max_value=4, value=0, step=1, key="live_cam_idx")
        with ctrl_col3:
            st.markdown(
                f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.7rem; color: {'#10b981' if run_live else '#64748b'}; text-align: right;'>"
                f"{'● SENSOR ACTIVE' if run_live else '○ SENSOR STANDBY'}"
                f"</div>",
                unsafe_allow_html=True,
            )

        if run_live:
            _run_realtime_webcam_loop(rec_service, curr_thresh, int(cam_idx))
        else:
            render_html(
                """
                <div style="background: #080b11; border: 1px dashed #242d3e; border-radius: 8px; padding: 2rem 1.5rem; text-align: center; margin: 1rem auto; max-width: 650px;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.4rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                        <span class="hw-led amber"></span> REAL-TIME SCANNER STANDBY
                    </div>
                    <div style="font-size: 0.8125rem; color: #94a3b8; font-family: 'Inter', sans-serif; line-height: 1.5; margin-bottom: 0.75rem;">
                        Click <strong>⚡ ENGAGE LIVE REAL-TIME OPTICAL SCANNER</strong> above to start streaming from your webcam.<br>
                        The system runs real-time YuNet localization and SFace matching at ~20 FPS.
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: #64748b;">
                        TARGET: DIRECTSHOW OPTICAL CH-00 | TRANSIENT QUERY MODE (ZERO GALLERY MUTATION)
                    </div>
                </div>
                """
            )

    # ---------------------------------------------------------------------
    # SUB-MODE B: HIGH-RES SNAPSHOT CAPTURE
    # ---------------------------------------------------------------------
    else:
        if "identify_cam_ver" not in st.session_state:
            st.session_state["identify_cam_ver"] = 0

        cam_key = f"identify_cam_{st.session_state['identify_cam_ver']}"
        cam_picture = st.camera_input(
            "Live Identification Camera Input",
            key=cam_key,
            label_visibility="collapsed",
            help="Align face in center viewfinder and click 'Take Photo' to initiate biometric matching.",
        )

        if cam_picture is not None:
            raw_bytes = cam_picture.getvalue()
            raw_bgr = load_image(raw_bytes)

            t0 = time.perf_counter()
            with st.spinner("Executing face localization, 5-point alignment, and gallery comparison..."):
                output = rec_service.identify(raw_bgr, threshold=curr_thresh, source="webcam")
            latency_ms = (time.perf_counter() - t0) * 1000.0

            _render_forensic_report(
                rec_service=rec_service,
                output=output,
                raw_bgr=raw_bgr,
                curr_thresh=curr_thresh,
                latency_ms=latency_ms,
                input_label="LIVE WEBCAM SNAPSHOT CAPTURE",
                active_source="webcam",
            )
        else:
            render_html(
                """
                <div style="background: #080b11; border: 1px dashed #242d3e; border-radius: 8px; padding: 1.5rem 1rem; text-align: center; margin: 1rem auto; max-width: 600px;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.25rem;">
                        ● BROWSER WEBCAM VIEWFINDER ENGAGED
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; font-family: 'Inter', sans-serif;">
                        Position your face inside the central reticle and click <strong>Take Photo</strong> on the camera display to run biometric identification.
                    </div>
                </div>
                """
            )


def _run_realtime_webcam_loop(
    rec_service: FaceRecognitionService,
    curr_thresh: float,
    cam_idx: int = 0,
):
    """Execute continuous real-time video stream facial identification."""
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)

    if not cap.isOpened():
        st.error(f"❌ Could not open optical camera at index {cam_idx}. Ensure camera is plugged in and permissions are granted.")
        return

    frame_placeholder = st.empty()
    telemetry_placeholder = st.empty()

    frame_count = 0
    t_start = time.time()

    try:
        while st.session_state.get("live_cam_toggle", False):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            t_frame_start = time.perf_counter()

            # Execute full pipeline: detection, embedding, matching
            output = rec_service.identify(frame, threshold=curr_thresh, source="webcam")
            frame_latency_ms = (time.perf_counter() - t_frame_start) * 1000.0

            total_elapsed = time.time() - t_start
            live_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0

            # Draw HUD telemetry banner directly on top of frame
            annotated_frame = output.annotated_image.copy()
            hud_bg = np.zeros((36, annotated_frame.shape[1], 3), dtype=np.uint8)
            hud_color = (0, 255, 0) if output.num_matches > 0 else ((0, 200, 255) if output.num_faces_detected > 0 else (128, 128, 128))
            hud_text = (
                f"FPS: {live_fps:.1f} | LATENCY: {frame_latency_ms:.0f}ms | "
                f"FACES: {output.num_faces_detected} | MATCH: {output.num_matches} | UNKNOWN: {output.num_unknowns} | "
                f"THR: {curr_thresh:.2f}"
            )
            cv2.putText(
                annotated_frame,
                hud_text,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                hud_color,
                2,
                cv2.LINE_AA,
            )

            # Display annotated live frame in Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)

            # Render lightweight telemetry bar below frame
            match_summary = ", ".join([f"{r.identity} ({r.similarity:.2f})" for r in output.results if r.accepted]) or "None"
            telemetry_placeholder.markdown(
                f"""
                <div style="background: #080b11; border: 1px solid #1c2536; border-radius: 6px; padding: 0.5rem 0.85rem; display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    <div>
                        <span style="color: #64748b;">ACTIVE MATCHES:</span>
                        <strong style="color: {'#10b981' if output.num_matches > 0 else '#94a3b8'}; margin-left: 0.3rem;">{match_summary}</strong>
                    </div>
                    <div style="display: flex; gap: 1rem; color: #64748b;">
                        <span>FPS: <strong style="color: #f8fafc;">{live_fps:.1f}</strong></span>
                        <span>DEVICE: <strong style="color: #f59e0b;">{rec_service.device_info}</strong></span>
                        <span style="color: #10b981; font-weight: 700;">● STREAMING</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Yield control for responsiveness
            time.sleep(0.01)

    finally:
        cap.release()


def _render_upload_identification_tab(
    rec_service: FaceRecognitionService,
    curr_thresh: float,
):
    """Render the static image file upload identification workstation."""
    if "identify_uploader_ver" not in st.session_state:
        st.session_state["identify_uploader_ver"] = 0

    uploader_key = f"identify_uploader_{st.session_state['identify_uploader_ver']}"
    query_file = st.file_uploader(
        "Acquire Optical Frame",
        type=["jpg", "jpeg", "png", "webp"],
        key=uploader_key,
        help="Select an optical image to engage the face detector and gallery search.",
    )

    if query_file is not None:
        raw_bytes = query_file.getvalue()
        raw_bgr = load_image(raw_bytes)

        t0 = time.perf_counter()
        with st.spinner("Executing face localization, 5-point alignment, and gallery comparison..."):
            output = rec_service.identify(raw_bgr, threshold=curr_thresh, source="upload")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        _render_forensic_report(
            rec_service=rec_service,
            output=output,
            raw_bgr=raw_bgr,
            curr_thresh=curr_thresh,
            latency_ms=latency_ms,
            input_label=f"FILE: {query_file.name}",
            active_source="upload",
        )
    else:
        render_empty_state(
            title="NO INPUT SIGNAL DETECTED",
            description="Acquire or upload an optical image to engage the face detector and compare against the enrolled biometric gallery.",
            icon_name="identify",
        )


def _render_forensic_report(
    rec_service: FaceRecognitionService,
    output: IdentificationOutput,
    raw_bgr: np.ndarray,
    curr_thresh: float,
    latency_ms: float,
    input_label: str,
    active_source: str,
):
    """Render the full biometric analysis workstation layout, decision panels, and telemetry."""
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Split-Screen Workstation Layout
    col_monitor, col_decision = st.columns([3, 2])

    with col_monitor:
        render_html(
            f"""
            <div class="camera-monitor-bezel">
                <div class="camera-corner-tag">
                    <span>[ SENSOR CH-01 // {input_label} ]</span>
                    <span>DETECTED: {output.num_faces_detected} | MATCH: {output.num_matches} | UNKNOWN: {output.num_unknowns}</span>
                </div>
            </div>
            """
        )

        # Annotated Image Viewer
        st.image(
            bgr_to_rgb(output.annotated_image),
            use_container_width=True,
        )

        # Action Toolbar below viewer
        act_col1, act_col2 = st.columns([1, 1])
        with act_col1:
            success, enc_img = cv2.imencode(".jpg", output.annotated_image)
            if success:
                st.download_button(
                    label="📥 EXPORT ANNOTATED FRAME",
                    data=enc_img.tobytes(),
                    file_name="vision_id_analysis_frame.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
        with act_col2:
            if active_source == "upload":
                if st.button("🔄 ACQUIRE ANOTHER FRAME", use_container_width=True, key="btn_another_frame"):
                    st.session_state["identify_uploader_ver"] = st.session_state.get("identify_uploader_ver", 0) + 1
                    st.rerun()
            else:
                if st.button("📷 🔄 RETAKE / NEW SCAN", type="primary", use_container_width=True, key="btn_retake_cam"):
                    st.session_state["identify_cam_ver"] = st.session_state.get("identify_cam_ver", 0) + 1
                    st.rerun()

    with col_decision:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led amber"></span> BIOMETRIC DECISION LOG
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                INDIVIDUAL SUBJECT CLASSIFICATION & TELEMETRY
            </div>
            """
        )

        if output.num_faces_detected == 0:
            render_html(
                """
                <div class="panel-plate" style="border-left: 4px solid #f59e0b;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; color: #f59e0b; font-weight: 700;">
                        ⚠ NO FACE PATTERN DETECTED
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                        The optical frame contains no locatable facial structures with sufficient confidence. Ensure frontal orientation and adequate illumination.
                    </div>
                </div>
                """
            )
        else:
            for res in output.results:
                is_match = res.accepted

                # Physical Decision Panel
                render_decision_panel(
                    face_idx=res.face_index,
                    identity=res.identity,
                    similarity=res.similarity,
                    threshold=curr_thresh,
                    accepted=is_match,
                    detection_confidence=res.detection_confidence,
                )

                # Thumbnail + Similarity Needle Gauge
                c_thumb, c_gauge = st.columns([1, 2], vertical_alignment="center")
                with c_thumb:
                    face_crop = crop_face(raw_bgr, res.bbox, margin=0.15)
                    crop_rgb = bgr_to_rgb(face_crop)
                    render_html("<div class='camera-monitor-bezel' style='padding: 0.35rem; margin-bottom: 0.5rem;'>")
                    st.image(crop_rgb, use_container_width=True)
                    render_html("</div>")
                with c_gauge:
                    render_similarity_gauge(res.similarity, curr_thresh, is_match)

                # Verification Trace
                render_analysis_trace(
                    detected=True,
                    aligned=True,
                    embedded=True,
                    best_similarity=res.similarity,
                    threshold=curr_thresh,
                    accepted=is_match,
                )

                # Candidate Rankings (Collapsible)
                if res.top_candidates:
                    with st.expander(f"Gallery Similarity Rankings [Face #{res.face_index + 1}]", expanded=False):
                        cand_data = []
                        for rank, cand in enumerate(res.top_candidates, 1):
                            cand_data.append(
                                {
                                    "Rank": f"#{rank:02d}",
                                    "Enrolled Identity": cand.person_name,
                                    "Cosine Similarity": f"{cand.similarity:.4f}",
                                    "Above Threshold": "✓ ACCEPT" if cand.similarity >= curr_thresh else "✗ REJECT",
                                }
                            )
                        st.dataframe(pd.DataFrame(cand_data), hide_index=True, use_container_width=True)

                # Deliberate Reference Addition (Section 18: Non-automatic, explicit confirmation)
                if is_match and res.person_id is not None and res.embedding is not None:
                    with st.expander(f"➕ ENROLLMENT EXTENSION: ADD AS REFERENCE [Face #{res.face_index + 1}]", expanded=False):
                        st.markdown(
                            f"""
                            <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'Inter', sans-serif; margin-bottom: 0.5rem; line-height: 1.4;">
                                This captured query frame matched <strong>{res.identity}</strong> with cosine similarity <strong>{res.similarity:.4f}</strong>.
                                You may deliberately register this vector as an additional reference in the gallery.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        btn_key = f"btn_add_ref_{res.face_index}_{res.person_id}_{int(time.time())}"
                        if st.button(
                            f"CONFIRM & ADD REFERENCE TO {res.identity.upper()}",
                            key=btn_key,
                            type="secondary",
                            use_container_width=True,
                        ):
                            try:
                                rec_service.db.add_embedding(
                                    person_id=res.person_id,
                                    embedding=res.embedding,
                                    source_image_name=f"ref_{active_source}_{int(time.time())}.jpg",
                                    source=active_source,
                                )
                                st.success(f"✓ Biometric reference vector successfully added to {res.identity}'s profile!")
                            except Exception as e:
                                st.error(f"Failed to add reference: {e}")

                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Collapsible Technical Pipeline Diagnostic Rack
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    with st.expander("SYSTEM DIAGNOSTICS & HARDWARE PIPELINE TELEMETRY ▾", expanded=False):
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            st.metric("DETECTOR ARCHITECTURE", config.detector_model_name.upper(), "YuNet ONNX (Dynamic)")
        with t_col2:
            st.metric("EMBEDDING MODEL", config.embedder_model_name.upper(), f"{config.embedding_dimension}-d ArcFace CNN")
        with t_col3:
            st.metric("PIPELINE LATENCY", f"{latency_ms:.1f} ms", f"{rec_service.device_info.upper()}")
        with t_col4:
            st.metric("DECISION METRIC", "COSINE SIMILARITY", f"τ = {curr_thresh:.2f}")
