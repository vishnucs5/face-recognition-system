"""Identification view: Biometric workstation, dual-mode optical acquisition (Upload & Webcam), decision panels, and rejection trace."""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.config import config
from app.recognition.recognizer import FaceRecognitionService
from app.ui.components import (
    render_analysis_trace,
    render_camera_permission_notice,
    render_capture_source_selector,
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
    """Render the physical identification workstation with Upload and Live Webcam Scanner modes."""
    curr_thresh = float(st.session_state.get("threshold", config.match_threshold))

    # Instrument Header
    render_hardware_header(
        page_title="IDENTIFY TERMINAL",
        subtitle="OPTICAL SIGNAL ACQUISITION, EMBEDDING MATCHING, & UNKNOWN REJECTION AUDIT",
        model_status="READY",
        device=rec_service.device_info,
        threshold=curr_thresh,
    )

    # Explicit Workflow Banner (Transient Query Mode)
    render_workflow_mode_badge("identification")

    # Initialize session state tracking
    if "identify_source_mode" not in st.session_state:
        st.session_state["identify_source_mode"] = "upload"
    if "identify_uploader_ver" not in st.session_state:
        st.session_state["identify_uploader_ver"] = 0
    if "identify_cam_ver" not in st.session_state:
        st.session_state["identify_cam_ver"] = 0

    active_source = st.session_state["identify_source_mode"]

    # =========================================================================
    # TOP TOOLBAR: Mode Selector Rockers + Operating Threshold Slider
    # =========================================================================
    top_col1, top_col2 = st.columns([3, 2], vertical_alignment="center")

    with top_col1:
        render_capture_source_selector(active_source, "ACQUISITION SOURCE SELECTOR")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(
                "📁 [01] IMAGE UPLOAD SCAN",
                type="primary" if active_source == "upload" else "secondary",
                use_container_width=True,
                key="btn_id_switch_upload",
            ):
                st.session_state["identify_source_mode"] = "upload"
                st.rerun()

        with col_btn2:
            if st.button(
                "📷 [02] WEBCAM LIVE SCANNER",
                type="primary" if active_source == "webcam" else "secondary",
                use_container_width=True,
                key="btn_id_switch_webcam",
            ):
                st.session_state["identify_source_mode"] = "webcam"
                st.rerun()

    with top_col2:
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

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    query_bytes = None
    input_label = ""

    # =========================================================================
    # MODE 1: IMAGE FILE UPLOAD
    # =========================================================================
    if active_source == "upload":
        uploader_key = f"identify_uploader_{st.session_state['identify_uploader_ver']}"
        query_file = st.file_uploader(
            "Acquire Optical Frame",
            type=["jpg", "jpeg", "png", "webp"],
            key=uploader_key,
            help="Select an optical image to engage the face detector and gallery search.",
        )
        if query_file is not None:
            query_bytes = query_file.getvalue()
            input_label = f"FILE: {query_file.name}"
        else:
            render_empty_state(
                title="NO INPUT SIGNAL DETECTED",
                description="Acquire or upload an optical image to engage the face detector and compare against the enrolled biometric gallery.",
                icon_name="identify",
            )
            return

    # =========================================================================
    # MODE 2: WEBCAM LIVE SCANNER
    # =========================================================================
    else:
        # Privacy & Camera Permission Header
        render_camera_permission_notice(is_denied=False)

        # Biometric Webcam Monitor Bezel & Alignment Reticle
        render_webcam_monitor_bezel(
            channel_name="CH-01 // LIVE SCANNER VIEWPORT",
            status_text="LIVE",
            led_color="green",
            device_label="INTEGRATED OPTICAL SENSOR",
        )
        render_face_alignment_guide()

        cam_key = f"identify_cam_{st.session_state['identify_cam_ver']}"
        cam_picture = st.camera_input(
            "Live Identification Camera Input",
            key=cam_key,
            label_visibility="collapsed",
            help="Align face in center viewfinder and click 'Take Photo' to initiate biometric matching.",
        )

        if cam_picture is not None:
            query_bytes = cam_picture.getvalue()
            input_label = "LIVE WEBCAM OPTICAL CAPTURE"
        else:
            render_html(
                """
                <div style="background: #080b11; border: 1px dashed #242d3e; border-radius: 8px; padding: 1.5rem 1rem; text-align: center; margin: 1rem auto; max-width: 600px;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.25rem;">
                        ● LIVE WEBCAM SCANNER ENGAGED
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; font-family: 'Inter', sans-serif;">
                        Position your face inside the central reticle and click <strong>Take Photo</strong> on the camera display to run real-time biometric identification.
                    </div>
                </div>
                """
            )
            return

    # =========================================================================
    # EXECUTE BIOMETRIC IDENTIFICATION PIPELINE
    # =========================================================================
    t0 = time.perf_counter()
    with st.spinner("Executing face localization, 5-point alignment, and gallery comparison..."):
        output = rec_service.identify(query_bytes, threshold=curr_thresh, source=active_source)
    latency_ms = (time.perf_counter() - t0) * 1000.0

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
                if st.button("🔄 ACQUIRE ANOTHER FRAME", use_container_width=True):
                    st.session_state["identify_uploader_ver"] += 1
                    st.rerun()
            else:
                if st.button("📷 🔄 RETAKE / NEW SCAN", type="primary", use_container_width=True):
                    st.session_state["identify_cam_ver"] += 1
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
            raw_bgr = load_image(query_bytes)
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
                        btn_key = f"btn_add_ref_{res.face_index}_{res.person_id}"
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
