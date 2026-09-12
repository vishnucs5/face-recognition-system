"""Evaluation analytics view: Biometric benchmarking, oscilloscope tradeoff display, and calibration tables."""

import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

from app.config import config
from app.evaluation.evaluator import EvaluationService
from app.ui.components import (
    render_empty_state,
    render_hardware_header,
    render_lcd_metric_gauge,
)
from app.ui.theme import ICONS, THEME, render_html


def render_evaluation_view(eval_service: EvaluationService):
    """Render the physical biometric performance evaluation bench of the VISION-ID console."""
    curr_thresh = float(st.session_state.get("threshold", config.match_threshold))

    # Instrument Header
    render_hardware_header(
        page_title="PERFORMANCE BENCHMARK",
        subtitle="BIOMETRIC SELECTIVITY, FAR/FRR TRADEOFF CURVES, & THRESHOLD CALIBRATION",
        model_status="BENCH READY",
        device="CPU",
        threshold=curr_thresh,
    )

    known_samples, unknown_samples = eval_service.scan_test_dataset()
    has_dataset = len(known_samples) > 0 or len(unknown_samples) > 0

    # Inset Dataset Telemetry Bar
    render_html(
        f"""
        <div class="panel-plate" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;">
            <div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #f8fafc;">
                    EVALUATION DATASET TELEMETRY
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem;">
                    KNOWN PROBE SAMPLES: <strong style="color: #10b981;">{len(known_samples):03d}</strong> | 
                    UNKNOWN IMPOSTER SAMPLES: <strong style="color: #ef4444;">{len(unknown_samples):03d}</strong>
                </div>
            </div>
            <div>
                <span class="hw-badge {'active-green' if has_dataset else 'active-amber'}">
                    <span class="hw-led {'green' if has_dataset else 'off'}"></span>
                    {'● DATASET LOADED' if has_dataset else '● NO DATASET FOUND'}
                </span>
            </div>
        </div>
        """
    )

    if not has_dataset:
        render_empty_state(
            title="NO CALIBRATION SIGNAL / TEST DATA",
            description="Evaluation requires labeled probe photographs in 'data/test/known/' and 'data/test/unknown/'.",
            icon_name="evaluation",
        )
        if st.button("⚡ GENERATE SYNTHETIC BENCHMARK SAMPLES", type="primary"):
            from scripts import prepare_sample_data
            with st.spinner("Synthesizing benchmark optical probes..."):
                prepare_sample_data.create_sample_dataset(enroll_identities=False)
            st.success("Benchmark dataset generated! Initiate sweep below.")
            st.rerun()
        return

    # Trigger Evaluation Button
    eval_btn = st.button("🚀 INITIATE THRESHOLD SWEEP (τ = 0.30 → 0.80)", type="primary")

    csv_path = config.eval_output_dir / "threshold_analysis.csv"
    plot_path = config.eval_output_dir / "threshold_analysis.png"
    report_json_path = config.eval_output_dir / "report.json"

    if eval_btn:
        with st.spinner("Sweeping thresholds across range and computing biometric confusion matrices..."):
            report = eval_service.evaluate()
        if report.dataset_available:
            st.success(report.summary_message)
        else:
            st.warning(report.summary_message)

    # Check if results exist
    if csv_path.exists():
        df_metrics = pd.read_csv(csv_path)

        # Telemetry metrics at operating threshold
        row_at_thresh = df_metrics.iloc[(df_metrics["threshold"] - curr_thresh).abs().argsort()[:1]].iloc[0]

        top_col1, top_col2, top_col3, top_col4 = st.columns(4)
        with top_col1:
            render_lcd_metric_gauge(
                label="SYSTEM ACCURACY",
                value=f"{row_at_thresh['accuracy'] * 100:.1f}%",
                subtext=f"At operating τ = {row_at_thresh['threshold']:.2f}",
                icon_name="check",
            )
        with top_col2:
            render_lcd_metric_gauge(
                label="FALSE ACCEPT (FAR)",
                value=f"{row_at_thresh['far'] * 100:.1f}%",
                subtext="Imposter breach vulnerability",
                icon_name="alert",
            )
        with top_col3:
            render_lcd_metric_gauge(
                label="FALSE REJECT (FRR)",
                value=f"{row_at_thresh['frr'] * 100:.1f}%",
                subtext="Legitimate user lockout rate",
                icon_name="alert",
            )
        with top_col4:
            render_lcd_metric_gauge(
                label="TRUE ACCEPT (TAR)",
                value=f"{row_at_thresh['tar'] * 100:.1f}%",
                subtext="Genuine recognition recall",
                icon_name="shield",
            )

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        # Oscilloscope Chart & Table
        col_plot, col_table = st.columns([1, 1])

        with col_plot:
            render_html(
                """
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span class="hw-led green"></span> OSCILLOSCOPE CH-02 // SELECTIVITY TRADEOFF
                </div>
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                    ACCURACY, FAR, &amp; FRR VS THRESHOLD CONTINUOUS SPECTRUM
                </div>
                """
            )
            if plot_path.exists():
                render_html("<div class='camera-monitor-bezel'>")
                im = Image.open(plot_path)
                st.image(im, use_container_width=True)
                render_html("</div>")
            render_html(
                f"""
                <div class="glass-display" style="margin-top: 0.75rem; padding: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #94a3b8; line-height: 1.5;">
                    💡 <strong style="color: {THEME['accent_amber']};">CALIBRATION PRINCIPLE:</strong> 
                    Raising threshold τ forces strict zero-tolerance imposter rejection (FAR &rarr; 0%), but increases genuine user lockout (FRR). Optimal biometric operating point balances both according to security profile.
                </div>
                """
            )

        with col_table:
            render_html(
                """
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span class="hw-led blue"></span> CALIBRATION SWEEP TELEMETRY TABLE
                </div>
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                    EMPIRICAL MEASUREMENTS RECORDED ACROSS RANGE
                </div>
                """
            )
            display_df = df_metrics[[
                "threshold", "tar", "far", "frr", "accuracy",
                "correct_known_accepts", "false_known_rejects",
                "false_unknown_accepts", "correct_unknown_rejects",
            ]].rename(columns={
                "threshold": "Threshold (τ)",
                "tar": "TAR",
                "far": "FAR",
                "frr": "FRR",
                "accuracy": "Accuracy",
                "correct_known_accepts": "Known Acc",
                "false_known_rejects": "Known Rej",
                "false_unknown_accepts": "Unk Acc",
                "correct_unknown_rejects": "Unk Rej",
            })
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            # Export Telemetry Buttons
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="📥 EXPORT CSV TELEMETRY",
                        data=f.read(),
                        file_name="vision_id_threshold_analysis.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with d_col2:
                if report_json_path.exists():
                    with open(report_json_path, "rb") as f:
                        st.download_button(
                            label="📥 EXPORT JSON REPORT",
                            data=f.read(),
                            file_name="vision_id_evaluation_report.json",
                            mime="application/json",
                            use_container_width=True,
                        )
    else:
        st.info("Execute 'INITIATE THRESHOLD SWEEP' above to compute biometric confusion metrics across the threshold spectrum.")
