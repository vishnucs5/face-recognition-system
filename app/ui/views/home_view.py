"""Home view: Central access portal to all modules in the Facial Recognition Console."""

import streamlit as st

from app.config import config
from app.database import DatabaseManager
from app.recognition.recognizer import FaceRecognitionService
from app.ui.components import (
    render_bios_boot_banner,
    render_hardware_header,
    render_lcd_metric_gauge,
)
from app.ui.theme import ICONS, THEME, render_html


def render_home_view(
    db_mgr: DatabaseManager,
    rec_service: FaceRecognitionService,
):
    """Render the central home portal with direct access to all biometric workstation modules."""
    threshold = float(st.session_state.get("threshold", config.match_threshold))
    identities_count = db_mgr.count_identities()
    embeddings_count = db_mgr.count_embeddings()

    # Hardware Instrument Top Header
    render_hardware_header(
        page_title="CENTRAL PORTAL",
        subtitle="FACIAL RECOGNITION WORKSTATION // ALL MODULES ACCESS & CONTROL HUB",
        model_status="SYSTEM READY",
        device=rec_service.device_info,
        threshold=threshold,
    )

    # Hardware BIOS / POST Self-Test Banner
    render_bios_boot_banner()

    # Master Console Hero Plate
    render_html(
        f"""
        <div class="panel-plate" style="margin-bottom: 1.5rem; padding: 1.5rem 1.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div class="hw-screw"></div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {THEME['accent_amber']}; letter-spacing: 0.12em; text-transform: uppercase;">
                        FACIAL RECOGNITION SYSTEM // MAIN CONTROL HUB
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #10b981; font-weight: 700;">
                        <span class="hw-led green"></span> ALL SUBSYSTEMS OPERATIONAL
                    </span>
                    <div class="hw-screw"></div>
                </div>
            </div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em; margin-bottom: 0.4rem;">
                Facial Recognition Biometric Console
            </div>
            <div style="font-size: 0.875rem; color: #94a3b8; max-width: 860px; line-height: 1.6; font-family: 'Inter', sans-serif;">
                High-precision deep biometric identification instrument. Features real-time multi-face localization via 
                <strong>YuNet ONNX</strong>, 5-point landmark affine normalization, 128-dimensional <strong>SFace / ArcFace</strong> feature extraction, 
                vectorized cosine gallery search, and an uncompromising <strong>unknown imposter rejection</strong> boundary.
            </div>
        </div>
        """
    )

    # Telemetry Status Band (4 LCD Segment Gauges)
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        render_lcd_metric_gauge(
            label="ENROLLED IDENTITIES",
            value=f"{identities_count:03d}",
            subtext="Subjects in biometric gallery",
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
            subtext="Cosine decision boundary",
            icon_name="evaluation",
        )
    with g_col4:
        render_lcd_metric_gauge(
            label="EXECUTION ENGINE",
            value=config.embedder_model_name.upper(),
            subtext=f"Target: {rec_service.device_info.upper()}",
            icon_name="cpu",
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Section Header: Workstation Access Hub
    render_html(
        """
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.08em; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.5rem;">
            <span class="hw-led amber"></span> WORKSTATION MODULE SELECTOR
        </div>
        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 1rem; font-family: 'JetBrains Mono', monospace;">
            DIRECT ACCESS TO ALL PRIMARY BIOMETRIC OPERATION MODULES
        </div>
        """
    )

    # Row 1: Primary Operations (Identify, Enroll)
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        render_html(
            f"""
            <div class="panel-plate" style="margin-bottom: 0.5rem; height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {THEME['accent_amber']};">
                            MODULE 01 // OPERATIONAL
                        </span>
                        <span class="hw-badge active-green" style="font-size: 0.65rem;">
                            <span class="hw-led green"></span> READY
                        </span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">
                        🎯 Identify Faces // Live Webcam &amp; Upload
                    </div>
                    <div style="font-size: 0.8125rem; color: #94a3b8; line-height: 1.5;">
                        Acquire query optical images via real-time live webcam scanner, browser snapshot camera, 
                        or file upload for deep embedding matching and unknown imposter rejection.
                    </div>
                </div>
            </div>
            """
        )
        if st.button("📷 ⚡ LAUNCH WEBCAM IDENTIFIER", type="primary", key="home_btn_identify", use_container_width=True):
            st.session_state["nav"] = "identify"
            st.rerun()

    with r1_col2:
        render_html(
            f"""
            <div class="panel-plate" style="margin-bottom: 0.5rem; height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {THEME['accent_amber']};">
                            MODULE 02 // INGESTION
                        </span>
                        <span class="hw-badge active-green" style="font-size: 0.65rem;">
                            <span class="hw-led green"></span> READY
                        </span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">
                        ➕ Personnel Registration &amp; QC Bench
                    </div>
                    <div style="font-size: 0.8125rem; color: #94a3b8; line-height: 1.5;">
                        Register subjects into local biometric database with automated quality gating: 
                        sharpness audit, single-face validation, resolution check, and pose limits.
                    </div>
                </div>
            </div>
            """
        )
        if st.button("➕ LAUNCH ENROLLMENT BENCH", key="home_btn_enroll", use_container_width=True):
            st.session_state["nav"] = "enroll"
            st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Row 2: Management & Live Telemetry (People Directory, Operations Deck)
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        render_html(
            f"""
            <div class="panel-plate" style="margin-bottom: 0.5rem; height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {THEME['accent_amber']};">
                            MODULE 03 // ARCHIVE
                        </span>
                        <span class="hw-badge" style="font-size: 0.65rem;">
                            {identities_count} RECORDS
                        </span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">
                        👥 Enrolled Personnel Registry
                    </div>
                    <div style="font-size: 0.8125rem; color: #94a3b8; line-height: 1.5;">
                        Search and inspect all registered identities in the hardware database. Slide-out 
                        inspection drawer to audit stored reference photographs and execute cascading deletions.
                    </div>
                </div>
            </div>
            """
        )
        if st.button("👥 OPEN PERSONNEL REGISTRY", key="home_btn_people", use_container_width=True):
            st.session_state["nav"] = "people"
            st.rerun()

    with r2_col2:
        render_html(
            f"""
            <div class="panel-plate" style="margin-bottom: 0.5rem; height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {THEME['accent_amber']};">
                            MODULE 04 // MONITORING
                        </span>
                        <span class="hw-badge active-amber" style="font-size: 0.65rem;">
                            TEST BAY ACTIVE
                        </span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">
                        📊 Operations Deck &amp; Optical Test Bay
                    </div>
                    <div style="font-size: 0.8125rem; color: #94a3b8; line-height: 1.5;">
                        Real-time biometric monitoring, quick optical drop-in sensor bay for instant identification, 
                        and complete subsystem architecture telemetry.
                    </div>
                </div>
            </div>
            """
        )
        if st.button("📊 OPEN OPERATIONS DECK", key="home_btn_dashboard", use_container_width=True):
            st.session_state["nav"] = "dashboard"
            st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Row 3: Evaluation & System Configuration (Evaluation Suite, System Settings)
    r3_col1, r3_col2 = st.columns(2)

    with r3_col1:
        render_html(
            f"""
            <div class="panel-plate" style="margin-bottom: 0.5rem; height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {THEME['accent_amber']};">
                            MODULE 05 // ANALYTICS
                        </span>
                        <span class="hw-badge active-green" style="font-size: 0.65rem;">
                            BENCHMARK READY
                        </span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">
                        📈 Performance Evaluation Suite
                    </div>
                    <div style="font-size: 0.8125rem; color: #94a3b8; line-height: 1.5;">
                        Biometric calibration workbench: sweep thresholds from 0.30 to 0.80, inspect continuous 
                        FAR/FRR tradeoff curves on Oscilloscope CH-02, and export reports in CSV/JSON.
                    </div>
                </div>
            </div>
            """
        )
        if st.button("📈 LAUNCH EVALUATION SUITE", key="home_btn_eval", use_container_width=True):
            st.session_state["nav"] = "evaluation"
            st.rerun()

    with r3_col2:
        render_html(
            f"""
            <div class="panel-plate" style="margin-bottom: 0.5rem; height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {THEME['accent_amber']};">
                            MODULE 06 // CONTROL
                        </span>
                        <span class="hw-badge" style="font-size: 0.65rem;">
                            AIR-GAPPED
                        </span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">
                        ⚙️ System Settings &amp; Governance
                    </div>
                    <div style="font-size: 0.8125rem; color: #94a3b8; line-height: 1.5;">
                        Hardware parameters: global decision threshold slider, multi-embedding reference aggregation 
                        policy (Max vs Mean), memory matrix cache, and air-gapped privacy guarantees.
                    </div>
                </div>
            </div>
            """
        )
        if st.button("⚙️ CONFIGURE SYSTEM", key="home_btn_settings", use_container_width=True):
            st.session_state["nav"] = "settings"
            st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Technical Pipeline Architecture Walkthrough Plate
    render_html(
        """
        <div class="panel-plate" style="padding: 1.25rem 1.5rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.08em; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led green"></span> END-TO-END PIPELINE ARCHITECTURE TRACE
            </div>
            <div class="glass-display" style="padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; line-height: 2;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span>[STEP 1] OPTICAL FRAME ACQUISITION</span>
                    <strong style="color: #f1f5f9;">640&times;640 RGB INPUT</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span>[STEP 2] FACE DETECTION &amp; LOCALIZATION</span>
                    <strong style="color: #38bdf8;">YuNet ONNX (0.6 CONFIDENCE BOUND)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span>[STEP 3] 5-POINT LANDMARK ALIGNMENT</span>
                    <strong style="color: #10b981;">EYES + NOSE + MOUTH CORNERS (AFFINE TRANSFORM)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span>[STEP 4] DEEP FEATURE EXTRACTION</span>
                    <strong style="color: #10b981;">SFACE RESIDUAL CNN &rarr; 128-D L2 UNIT VECTOR</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span>[STEP 5] GALLERY SIMILARITY SEARCH</span>
                    <strong style="color: #f59e0b;">COSINE SIMILARITY (NORMALIZED DOT PRODUCT)</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>[STEP 6] HARDWARE DECISION GATE</span>
                    <strong style="color: #ef4444;">SIM &ge; &tau; &rarr; MATCH  |  SIM &lt; &tau; &rarr; STRICT UNKNOWN REJECT</strong>
                </div>
            </div>
        </div>
        """
    )
