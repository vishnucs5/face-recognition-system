"""Enrollment view: Personnel registration console, biometric quality sensors, and sequence validation."""

import streamlit as st

from app.config import config
from app.database import DatabaseManager
from app.detection.detector import FaceDetector
from app.enrollment.service import EnrollmentService
from app.ui.components import (
    render_hardware_header,
    render_hardware_stepper,
)
from app.ui.theme import ICONS, THEME, render_html
from app.utils.image_utils import bgr_to_rgb, crop_face, draw_detection_annotations, load_image


def render_enroll_view(
    enroll_service: EnrollmentService,
    db_mgr: DatabaseManager,
):
    """Render the physical enrollment workstation of the VISION-ID console."""
    render_hardware_header(
        page_title="PERSONNEL REGISTRATION",
        subtitle="BIOMETRIC PROFILE ENROLLMENT & QUALITY VALIDATION BENCH",
        model_status="READY",
        device="CPU",
    )

    # 4-Stage Hardware Stepper
    step_names = ["SUBJECT ID", "OPTICAL INPUT", "QUALITY AUDIT", "DATABASE COMMIT"]
    current_step = 0

    # Layout: Form on Left, Optical Preview & Quality Sensors on Right
    col_form, col_sensor = st.columns([3, 2])

    with col_form:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led amber"></span> SUBJECT DOSSIER FORM
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                SPECIFY SUBJECT DESIGNATION AND REFERENCE OPTICAL FRAME
            </div>
            """
        )

        name_input = st.text_input(
            "Subject Identification / Full Name",
            placeholder="e.g. Alice Smith (ID: AS-001)",
            help="Alphanumeric identifier for the subject being enrolled.",
        )
        if name_input.strip():
            current_step = 1

        enroll_file = st.file_uploader(
            "Reference Optical Frame (Image)",
            type=["jpg", "jpeg", "png", "webp"],
            key="enroll_view_uploader",
        )
        if enroll_file is not None:
            current_step = 2

        skip_qc = st.checkbox(
            "Bypass biometric sharpness & resolution gates (Override)",
            value=False,
            help="Bypass Laplacian blur variance and minimum bounding box checks.",
        )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        enroll_submit = st.button("➕ COMMIT TO DATABASE", type="primary", use_container_width=True)

    with col_sensor:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led blue"></span> BIOMETRIC QUALITY SENSORS
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                AUTOMATED PRE-INGESTION HARDWARE CHECKS
            </div>
            """
        )

        render_html(
            f"""
            <div class="glass-display" style="margin-bottom: 1rem; padding: 1rem 1.25rem;">
                <div style="display: flex; flex-direction: column; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> SINGLE SUBJECT: Exactly 1 face per frame
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> OPTICAL RESOLUTION: Face bounding box &ge; 60x60 px
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> SHARPNESS AUDIT: Laplacian variance &ge; 25.0
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> POSE ORIENTATION: Frontal yaw &amp; pitch &lt; 45°
                    </div>
                </div>
            </div>
            """
        )

        # Real-Time Image Preview Monitor
        if enroll_file is not None:
            try:
                img_bytes = enroll_file.getvalue()
                raw_bgr = load_image(img_bytes)
                dets = enroll_service.detector.detect(raw_bgr)

                render_html(
                    f"""
                    <div class="camera-monitor-bezel">
                        <div class="camera-corner-tag">
                            <span>[ PRE-INGESTION SENSOR VIEW ]</span>
                            <span>FACES LOCALIZED: {len(dets)}</span>
                        </div>
                    </div>
                    """
                )

                if len(dets) == 1:
                    annotated = draw_detection_annotations(raw_bgr, dets)
                    st.image(bgr_to_rgb(annotated), caption="Single face localized with 5 facial landmark anchors", use_container_width=True)
                elif len(dets) == 0:
                    st.image(bgr_to_rgb(raw_bgr), use_container_width=True)
                    st.warning("⚠ SENSOR WARNING: No face detected in optical frame. Provide clear frontal view.")
                else:
                    annotated = draw_detection_annotations(raw_bgr, dets)
                    st.image(bgr_to_rgb(annotated), use_container_width=True)
                    st.error(f"⚠ SINGLE-FACE VIOLATION: {len(dets)} faces detected. Only 1 individual allowed per reference frame.")
            except Exception as e:
                st.error(f"Optical frame decoding error: {e}")

    # Top Sequence Stepper
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    render_hardware_stepper(step_names, current_step)

    # Handle Submission
    if enroll_submit:
        if not name_input.strip():
            st.error("Please enter a subject identifier before committing.")
        elif enroll_file is None:
            st.error("Please provide a reference optical frame.")
        else:
            with st.spinner("Executing localization, 5-point alignment, embedding generation, and database commit..."):
                result = enroll_service.enroll(
                    name=name_input.strip(),
                    image=enroll_file.getvalue(),
                    allow_save_image=True,
                    skip_quality_check=skip_qc,
                )

            if result.success:
                st.session_state["last_enrolled_name"] = result.person_name
                render_html(
                    f"""
                    <div class="panel-plate" style="border-left: 4px solid #10b981; text-align: center; padding: 1.5rem; margin-top: 1rem;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.35rem;">
                            <span class="hw-led green"></span> BIOMETRIC ENROLLMENT CONFIRMED
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; color: #94a3b8; margin-bottom: 1rem;">
                            SUBJECT: <strong style="color: #f1f5f9;">{result.person_name}</strong> | 
                            STORED REFERENCE EMBEDDINGS: <strong style="color: {THEME['accent_amber']};">{result.total_embeddings_for_person}</strong>
                        </div>
                    </div>
                    """
                )
                sc_col1, sc_col2 = st.columns([1, 1])
                with sc_col1:
                    if st.button("⚡ ENGAGE IDENTIFICATION TEST", use_container_width=True):
                        st.session_state["nav"] = "identify"
                        st.rerun()
                with sc_col2:
                    if st.button("👥 VIEW ENROLLED PERSONNEL", use_container_width=True):
                        st.session_state["nav"] = "people"
                        st.rerun()
            else:
                st.error(f"❌ ENROLLMENT REJECTED: {result.message}")
                if result.quality and result.quality.issues:
                    st.warning("Quality check failures: " + "; ".join(result.quality.issues))
