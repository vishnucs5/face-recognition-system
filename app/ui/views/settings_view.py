"""Settings view: Hardware configuration, runtime diagnostics, and air-gapped privacy notices."""

import streamlit as st

from app.config import config
from app.database import DatabaseManager
from app.ui.components import render_hardware_header
from app.ui.theme import ICONS, THEME, render_html


def render_settings_view(db_mgr: DatabaseManager):
    """Render the physical hardware configuration and system diagnostics rack."""
    curr_thresh = float(st.session_state.get("threshold", config.match_threshold))

    render_hardware_header(
        page_title="SYSTEM CONFIGURATION",
        subtitle="BIOMETRIC ENGINE CALIBRATION, RUNTIME TELEMETRY, & AIR-GAPPED PRIVACY",
        model_status="ONLINE",
        device="CONFIG",
        threshold=curr_thresh,
    )

    col_engine, col_hw = st.columns([1, 1])

    with col_engine:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led amber"></span> BIOMETRIC ENGINE PARAMETERS
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                INFERENCE SPECIFICATIONS &amp; MATCHING STRATEGIES
            </div>
            """
        )

        render_html(
            f"""
            <div class="panel-plate" style="padding: 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748b; margin-bottom: 0.2rem;">
                    ACTIVE FEATURE EXTRACTOR
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 800; color: #f8fafc;">
                    {config.embedder_model_name.upper()} (128-D VECTOR)
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {THEME['accent_amber']}; margin-top: 0.25rem;">
                    DEEP RESIDUAL CNN (SPHEREFACE2 / ARCFACE)
                </div>
            </div>
            """
        )

        new_thresh = st.slider(
            "Global Matching Decision Boundary (τ)",
            min_value=0.10,
            max_value=0.90,
            value=curr_thresh,
            step=0.01,
            help="Scores above this boundary are accepted as MATCH. Scores below are rejected as UNKNOWN.",
        )
        if new_thresh != curr_thresh:
            st.session_state["threshold"] = new_thresh

        new_strat = st.radio(
            "Multi-Embedding Reference Aggregation Policy",
            options=["max", "mean"],
            index=0 if config.matching_strategy == "max" else 1,
            format_func=lambda x: "MAXIMUM SIMILARITY (Best single reference match)" if x == "max" else "MEAN SIMILARITY (Centroid average across all references)",
            help="Mathematical strategy used when scoring a query vector against multiple stored references of an enrolled identity.",
        )

    with col_hw:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led blue"></span> RUNTIME SUBSYSTEM TELEMETRY
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                HARDWARE BUS &amp; DATABASE ENGINE INTEGRITY
            </div>
            """
        )

        render_html(
            f"""
            <div class="glass-display" style="padding: 1.25rem; margin-bottom: 1.25rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; line-height: 2;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span style="color: #94a3b8;">INFERENCE TARGET:</span>
                    <strong style="color: #38bdf8;">OPENCV DNN (CPU OPTIMIZED)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span style="color: #94a3b8;">DATABASE ENGINE:</span>
                    <strong style="color: #f8fafc;">SQLITE 3 (WAL MODE ACTIVE)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #141b27;">
                    <span style="color: #94a3b8;">MEMORY MATRIX CACHE:</span>
                    <strong style="color: #10b981;">● ONLINE (VECTORIZED DOT PRODUCT)</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">PERSISTENT FILE:</span>
                    <strong style="color: {THEME['accent_amber']};">{config.database_path.name}</strong>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #ef4444; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.4rem;">
                {ICONS['alert']} DANGER ZONE: GALLERY PURGE
            </div>
            """
        )

        if st.checkbox("Engage master safety interlock (Clear Database)"):
            st.warning("⚠ WARNING: This action irrevocably erases all enrolled personnel and embeddings.")
            if st.button("🚨 EXECUTE COMPLETE GALLERY PURGE", type="primary"):
                db_mgr.clear_all()
                st.success("All gallery records and embedding matrices erased.")
                st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Privacy & Legal Notices
    render_html(
        """
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
            <span class="hw-led green"></span> AIR-GAPPED PRIVACY &amp; BIOMETRIC GOVERNANCE
        </div>
        """
    )
    render_html(
        f"""
        <div class="panel-plate" style="padding: 1.25rem; font-size: 0.8125rem; color: #94a3b8; line-height: 1.6; font-family: 'Inter', sans-serif;">
            <strong style="color: #f8fafc; font-family: 'JetBrains Mono', monospace;">1. AIR-GAPPED LOCAL COMPUTE:</strong>
            All optical acquisition, face detection, 5-point alignment, embedding extraction, and cosine matching operations execute strictly on the local host machine. No biometric vectors or imagery are transmitted across external networks or third-party cloud APIs.<br><br>
            <strong style="color: #f8fafc; font-family: 'JetBrains Mono', monospace;">2. UNKNOWN IMPOSTER REJECTION INTEGRITY:</strong>
            The system strictly enforces the calibrated threshold boundary &tau;. Unenrolled query probes falling below &tau; are classified as UNKNOWN, guaranteeing that no closest-neighbor identity is leaked.<br><br>
            <strong style="color: #f8fafc; font-family: 'JetBrains Mono', monospace;">3. BIOMETRIC DATA INTEGRITY:</strong>
            Identities and embedding vectors are maintained in a secure local SQLite relational database with foreign-key cascade constraints. Removing a subject completely purges all associated biometric reference vectors from memory and storage.
        </div>
        """
    )
