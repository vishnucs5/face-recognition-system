"""VISION-ID: Physical Biometric Analysis Workstation & Instrument Console."""

from pathlib import Path
import streamlit as st

from app.config import config
from app.database import DatabaseManager
from app.enrollment.service import EnrollmentService
from app.evaluation.evaluator import EvaluationService
from app.recognition.recognizer import FaceRecognitionService
from app.ui.components import render_hardware_status_bar
from app.ui.theme import ICONS, THEME, get_global_css, render_html
from app.ui.views import (
    render_home_view,
    render_dashboard_view,
    render_enroll_view,
    render_evaluation_view,
    render_identify_view,
    render_people_view,
    render_settings_view,
)

# Instrument Console Page Configuration
st.set_page_config(
    page_title="Facial Recognition // Biometric Console",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject centralized skeuomorphic design tokens & styles
st.markdown(get_global_css(), unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Instantiate and cache core biometric services."""
    db_mgr = DatabaseManager()
    rec_service = FaceRecognitionService(db_manager=db_mgr)
    enroll_service = EnrollmentService(
        db_manager=db_mgr,
        detector=rec_service.detector,
        embedder=rec_service.embedder,
    )
    eval_service = EvaluationService(recognition_service=rec_service)
    return db_mgr, rec_service, enroll_service, eval_service


def main():
    # Initialize session state keys
    if "nav" not in st.session_state:
        st.session_state["nav"] = "home"
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = config.match_threshold

    db_mgr, rec_service, enroll_service, eval_service = get_services()

    # =========================================================================
    # SIDEBAR: Physical Rack Control Panel
    # =========================================================================
    with st.sidebar:
        # Hardware Unit Plate & Power Lamp
        render_html(
            f"""
            <div style="background: linear-gradient(180deg, #1c222e 0%, #121620 100%); border: 1px solid #283344; border-radius: 6px; padding: 1rem 0.85rem; margin-top: 0.5rem; margin-bottom: 1.25rem; box-shadow: 0 4px 12px rgba(0,0,0,0.6), inset 1px 1px 0 rgba(255,255,255,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div class="hw-screw"></div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #64748b; letter-spacing: 0.08em;">
                        FR-8820 // RACK 01
                    </span>
                    <div class="hw-screw"></div>
                </div>
                <div style="text-align: center; padding: 0.25rem 0;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 1.05rem; letter-spacing: 0.04em; color: #f8fafc; text-shadow: 0 1px 3px rgba(0,0,0,0.8); line-height: 1.2;">
                        FACIAL RECOGNITION
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {THEME['accent_amber']}; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.15rem;">
                        BIOMETRIC CONSOLE
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem; padding-top: 0.45rem; border-top: 1px solid #1c2536;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #94a3b8;">
                        <span class="hw-led green"></span> POWER BUS
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #10b981; font-weight: 700;">
                        ● ONLINE
                    </span>
                </div>
            </div>
            """
        )

        # Navigation Rocker Switches
        nav_items = [
            ("home", "Home Portal", "🏠"),
            ("identify", "Identify Face", "🎯"),
            ("enroll", "Enroll Person", "➕"),
            ("people", "People Directory", "👥"),
            ("dashboard", "Operations Deck", "📊"),
            ("evaluation", "Evaluation Suite", "📈"),
            ("settings", "System Settings", "⚙️"),
        ]

        active_nav = st.session_state["nav"]

        for nav_key, nav_title, nav_icon in nav_items:
            is_active = active_nav == nav_key
            btn_style = "primary" if is_active else "secondary"
            indicator = "●" if is_active else "○"
            btn_label = f"{indicator}   {nav_title}"

            if st.button(
                btn_label,
                key=f"nav_btn_{nav_key}",
                type=btn_style,
                use_container_width=True,
            ):
                st.session_state["nav"] = nav_key
                st.rerun()

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Inset LCD Status Panel
        identities_count = db_mgr.count_identities()
        embeddings_count = db_mgr.count_embeddings()
        curr_thresh = float(st.session_state["threshold"])

        render_html(
            f"""
            <div class="glass-display" style="margin-bottom: 1.25rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 700; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.45rem; display: flex; justify-content: space-between;">
                    <span>GALLERY STATUS</span>
                    <span class="hw-led green"></span>
                </div>
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">
                    <span>IDENTITIES:</span>
                    <strong style="color: {THEME['accent_amber']};">{identities_count:03d}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">
                    <span>EMBEDDINGS:</span>
                    <strong style="color: {THEME['accent_amber']};">{embeddings_count:03d}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8;">
                    <span>STORAGE:</span>
                    <strong style="color: #10b981;">SQLITE WAL</strong>
                </div>
            </div>
            """
        )

        # Calibrated Threshold Rotary/Slider Control
        st.markdown(
            f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.25rem;">
                CALIBRATED THRESHOLD (τ)
            </div>
            """,
            unsafe_allow_html=True,
        )

        new_thresh = st.slider(
            "Threshold (τ)",
            min_value=0.10,
            max_value=0.90,
            value=curr_thresh,
            step=0.01,
            label_visibility="collapsed",
            key="sidebar_thresh_slider",
            help="Global decision threshold. Faces with cosine similarity >= τ are accepted as MATCH; below are rejected as UNKNOWN.",
        )
        if new_thresh != curr_thresh:
            st.session_state["threshold"] = new_thresh

        render_html(
            f"""
            <div style="background: #05070a; border: 1px solid #1a2232; border-radius: 4px; padding: 0.4rem 0.6rem; display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748b; margin-top: 0.35rem; margin-bottom: 0.75rem;">
                <span>LOW [0.10]</span>
                <span style="color: {THEME['accent_amber']}; font-weight: 700;">VAL: {new_thresh:.2f}</span>
                <span>HIGH [0.90]</span>
            </div>
            <div style="font-size: 0.6875rem; color: #64748b; line-height: 1.4; font-family: 'JetBrains Mono', monospace;">
                • LOWER τ: HIGHER FAR RISK<br>
                • HIGHER τ: HIGHER FRR RISK
            </div>
            """
        )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: #475569; text-align: center; border-top: 1px solid #161c28; padding-top: 0.75rem;">
                FACIAL RECOGNITION CONSOLE • v2.4<br>
                YUNET + SFACE ONNX ENGINE
            </div>
            """
        )

    # =========================================================================
    # MAIN ROUTER: Dispatch to Active View
    # =========================================================================
    active_page = st.session_state["nav"]

    if active_page == "home":
        render_home_view(db_mgr, rec_service)
    elif active_page == "dashboard":
        render_dashboard_view(db_mgr, rec_service)
    elif active_page == "identify":
        render_identify_view(rec_service)
    elif active_page == "enroll":
        render_enroll_view(enroll_service, db_mgr)
    elif active_page == "people":
        render_people_view(db_mgr)
    elif active_page == "evaluation":
        render_evaluation_view(eval_service)
    elif active_page == "settings":
        render_settings_view(db_mgr)
    else:
        render_home_view(db_mgr, rec_service)

    # =========================================================================
    # BOTTOM CONSOLE STATUS STRIP
    # =========================================================================
    render_hardware_status_bar(
        detector_name=config.detector_model_name,
        embedder_name=config.embedder_model_name,
        device_info=rec_service.device_info,
        threshold=float(st.session_state["threshold"]),
        identities_count=identities_count,
        embeddings_count=embeddings_count,
    )


if __name__ == "__main__":
    main()
