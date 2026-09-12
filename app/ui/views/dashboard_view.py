"""Dashboard view: Instrument operations deck, LCD telemetry gauges, and quick optical test bay."""

import time
import streamlit as st

from app.config import config
from app.database import DatabaseManager
from app.recognition.recognizer import FaceRecognitionService
from app.ui.components import (
    render_bios_boot_banner,
    render_hardware_header,
    render_lcd_metric_gauge,
    render_similarity_gauge,
)
from app.ui.theme import ICONS, THEME, render_html
from app.utils.image_utils import bgr_to_rgb, crop_face, load_image


def render_dashboard_view(
    db_mgr: DatabaseManager,
    rec_service: FaceRecognitionService,
):
    """Render the main operations deck of the VISION-ID console."""
    threshold = float(st.session_state.get("threshold", config.match_threshold))
    identities_count = db_mgr.count_identities()
    embeddings_count = db_mgr.count_embeddings()

    # Physical Instrument Header
    render_hardware_header(
        page_title="OPERATIONS DECK",
        subtitle="REAL-TIME BIOMETRIC RECOGNITION & SUBSYSTEM TELEMETRY",
        model_status="SYSTEM READY",
        device=rec_service.device_info,
        threshold=threshold,
    )

    # Hardware BIOS / POST Banner
    render_bios_boot_banner()

    # Machined Chassis Control Deck Banner
    hero_col1, hero_col2 = st.columns([3, 1], vertical_alignment="center")
    with hero_col1:
        render_html(
            f"""
            <div class="panel-plate">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {THEME['accent_amber']}; letter-spacing: 0.1em; text-transform: uppercase;">
                        {ICONS['shield']} PRIMARY BIOMETRIC INSTRUMENT
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #64748b;">
                        STATION: LAB-OPS-01
                    </span>
                </div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em; margin-bottom: 0.35rem;">
                    Facial Recognition Operations Deck
                </div>
                <div style="font-size: 0.8125rem; color: #94a3b8; line-height: 1.5; font-family: 'Inter', sans-serif;">
                    Real-time multi-subject face detection, 5-point landmark alignment, 128-dimensional deep feature extraction,
                    and calibrated cosine similarity matching with strict imposter unknown rejection.
                </div>
            </div>
            """
        )
    with hero_col2:
        if st.button("⚡ IDENTIFY FACE", type="primary", use_container_width=True):
            st.session_state["nav"] = "identify"
            st.rerun()
        if st.button("➕ ENROLL SUBJECT", use_container_width=True):
            st.session_state["nav"] = "enroll"
            st.rerun()

    # Inset LCD Digital Readout Gauges (4 Instrument Displays)
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        render_lcd_metric_gauge(
            label="ENROLLED IDENTITIES",
            value=f"{identities_count:03d}",
            subtext="Enrolled subjects in database",
            icon_name="people",
        )
    with g_col2:
        render_lcd_metric_gauge(
            label="STORED EMBEDDINGS",
            value=f"{embeddings_count:03d}",
            subtext="128-d reference vectors",
            icon_name="database",
        )
    with g_col3:
        render_lcd_metric_gauge(
            label="MATCH THRESHOLD (τ)",
            value=f"{threshold:.2f}",
            subtext="Calibrated cosine boundary",
            icon_name="evaluation",
        )
    with g_col4:
        render_lcd_metric_gauge(
            label="BIOMETRIC ENGINE",
            value="SFACE",
            subtext="128-d ArcFace / ONNX",
            icon_name="cpu",
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Operations Workstation: Optical Sensor Test Bay + Telemetry
    col_sensor, col_telemetry = st.columns([3, 2])

    with col_sensor:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led amber"></span> OPTICAL SENSOR TEST BAY
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                ACQUIRE IMAGE FOR INSTANT REAL-TIME IDENTIFICATION & REJECTION AUDIT
            </div>
            """
        )

        test_img_file = st.file_uploader(
            "Input Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="dashboard_quick_uploader",
            label_visibility="collapsed",
        )

        if test_img_file is not None:
            img_bytes = test_img_file.getvalue()
            t0 = time.perf_counter()
            with st.spinner("Processing optical frame through detection & recognition pipeline..."):
                output = rec_service.identify(img_bytes, threshold=threshold)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            render_html(
                f"""
                <div class="camera-monitor-bezel">
                    <div class="camera-corner-tag">
                        <span>[ SENSOR CH-01 // OPTICAL MONITOR ]</span>
                        <span>LATENCY: {latency_ms:.1f}ms | TARGET: {rec_service.device_info}</span>
                    </div>
                </div>
                """
            )
            st.image(bgr_to_rgb(output.annotated_image), use_container_width=True)

            render_html(
                f"""
                <div style="background: #05070a; border: 1px solid #1a2232; border-radius: 4px; padding: 0.6rem 0.85rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; display: flex; justify-content: space-between; margin-top: 0.75rem;">
                    <span>DETECTED: <strong style="color: #f1f5f9;">{output.num_faces_detected}</strong></span>
                    <span>ACCEPTED: <strong style="color: #10b981;">{output.num_matches}</strong></span>
                    <span>REJECTED (UNKNOWN): <strong style="color: #ef4444;">{output.num_unknowns}</strong></span>
                </div>
                """
            )
        else:
            render_html(
                """
                <div class="camera-monitor-bezel" style="padding: 2.5rem 1rem; text-align: center;">
                    <div class="camera-corner-tag">
                        <span>[ SENSOR CH-01 // STANDBY ]</span>
                        <span>SIGNAL: AWAITING OPTICAL INPUT</span>
                    </div>
                    <div style="color: #475569; font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; margin-top: 1.5rem;">
                        NO INPUT SIGNAL DETECTED<br>
                        LOAD IMAGE TO ENGAGE RECOGNITION ENGINE
                    </div>
                </div>
                """
            )

    with col_telemetry:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led blue"></span> SYSTEM ARCHITECTURE TELEMETRY
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                SUBSYSTEM INTEGRITY & HARDWARE SPECIFICATIONS
            </div>
            """
        )

        render_html(
            f"""
            <div class="glass-display" style="padding: 1.25rem;">
                <div style="display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid #141b27; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    <span style="color: #94a3b8;">FACE DETECTOR:</span>
                    <span style="color: #10b981; font-weight: 700;"><span class="hw-led green"></span> {config.detector_model_name.upper()}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid #141b27; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    <span style="color: #94a3b8;">FEATURE EXTRACTOR:</span>
                    <span style="color: #10b981; font-weight: 700;"><span class="hw-led green"></span> {config.embedder_model_name.upper()} (128-D)</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid #141b27; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    <span style="color: #94a3b8;">DATABASE ENGINE:</span>
                    <span style="color: #f1f5f9; font-weight: 700;">SQLITE 3 (WAL MODE)</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid #141b27; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    <span style="color: #94a3b8;">INFERENCE TARGET:</span>
                    <span style="color: #38bdf8; font-weight: 700;">OPENCV DNN ({rec_service.device_info.upper()})</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid #141b27; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    <span style="color: #94a3b8;">DISTANCE METRIC:</span>
                    <span style="color: #f1f5f9; font-weight: 700;">COSINE SIMILARITY</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.45rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    <span style="color: #94a3b8;">UNKNOWN REJECTION:</span>
                    <span style="color: #f59e0b; font-weight: 700;"><span class="hw-led amber"></span> STRICT (τ = {threshold:.2f})</span>
                </div>
            </div>
            """
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_html(
            """
            <div class="panel-plate" style="padding: 1rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.35rem;">
                    OPERATING PROTOCOL
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.5; font-family: 'Inter', sans-serif;">
                    1. Query image is aligned using 5 detected facial landmarks.<br>
                    2. Vector is normalized to unit Euclidean sphere ($||x||_2 = 1$).<br>
                    3. Faces scoring below operating threshold τ are strictly classified as UNKNOWN without leaking candidate identity.
                </div>
            </div>
            """
        )
