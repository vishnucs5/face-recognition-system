"""Skeuomorphic hardware component library for VISION-ID Biometric Analysis Console."""

import base64
from typing import Dict, List, Optional
import cv2
import numpy as np
import streamlit as st

from app.ui.theme import ICONS, THEME, clean_html, render_html


def render_hardware_header(
    page_title: str,
    subtitle: str,
    model_status: str = "ONLINE",
    device: str = "CPU",
    threshold: Optional[float] = None,
):
    """Render physical instrument top command strip with hardware unit ID, firmware, and LED indicators."""
    is_online = "online" in model_status.lower() or "ready" in model_status.lower() or "active" in model_status.lower()
    led_class = "green" if is_online else "red"

    thresh_badge = (
        f"""<div class="hw-badge active-amber">
            <span>τ CALIBRATION:</span> <strong>{threshold:.2f}</strong>
        </div>"""
        if threshold is not None
        else ""
    )

    html = f"""
    <div class="top-instrument-bar" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap;">
            <div class="hw-screw"></div>
            <div>
                <h1 class="instrument-title" style="margin: 0; line-height: 1.2;">
                    <span style="color: {THEME['accent_amber']}; font-family: 'JetBrains Mono', monospace;">[FR-ID]</span>
                    {page_title}
                </h1>
                <p class="instrument-subtitle" style="margin: 0.2rem 0 0 0;">{subtitle}</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; justify-content: flex-end;">
            {thresh_badge}
            <div class="hw-badge active-green">
                <span class="hw-led {led_class}"></span> {model_status.upper()}
            </div>
            <div class="hw-badge">
                {ICONS['cpu']} {device.upper()}
            </div>
            <div class="hw-badge" style="color: #64748b;">
                HW: FR-8820
            </div>
            <div class="hw-screw"></div>
        </div>
    </div>
    """
    render_html(html)


def render_top_header(
    page_title: str,
    subtitle: str,
    model_status: str = "ONLINE",
    device: str = "CPU",
    threshold: Optional[float] = None,
):
    """Backwards-compatible alias for render_hardware_header."""
    render_hardware_header(page_title, subtitle, model_status, device, threshold)


def render_lcd_metric_gauge(
    label: str,
    value: str,
    subtext: str,
    icon_name: str = "dashboard",
):
    """Render inset digital LCD instrument gauge with engraved title and phosphor amber readout."""
    icon_svg = ICONS.get(icon_name, "")
    html = f"""
    <div class="lcd-metric-gauge">
        <div class="lcd-gauge-label">
            <span>{label}</span>
            <span>{icon_svg}</span>
        </div>
        <div class="lcd-gauge-value">{value}</div>
        <div class="lcd-gauge-subtext">{subtext}</div>
    </div>
    """
    render_html(html)


def render_metric_card(
    label: str,
    value: str,
    subtext: str,
    icon_name: str = "dashboard",
):
    """Backwards-compatible alias for render_lcd_metric_gauge."""
    render_lcd_metric_gauge(label, value, subtext, icon_name)


def render_similarity_gauge(
    similarity: float,
    threshold: float,
    accepted: bool,
):
    """Render physical analog-style bar gauge with illuminated threshold needle and margin readout."""
    sim_clamped = max(0.0, min(1.0, float(similarity)))
    thresh_clamped = max(0.0, min(1.0, float(threshold)))
    sim_percent = sim_clamped * 100.0
    thresh_percent = thresh_clamped * 100.0

    margin = similarity - threshold
    margin_str = f"+{margin:.3f}" if margin >= 0 else f"{margin:.3f}"
    margin_color = THEME["status_green"] if margin >= 0 else THEME["status_red"]
    status_fill = "match" if accepted else "unknown"

    html = f"""
    <div class="hw-sim-meter">
        <div class="hw-meter-labels">
            <span>SIMILARITY: <strong style="color: {'#10b981' if accepted else '#ef4444'}; font-family: 'JetBrains Mono', monospace;">{similarity:.4f}</strong></span>
            <span>MARGIN (Δ): <strong style="color: {margin_color}; font-family: 'JetBrains Mono', monospace;">{margin_str}</strong></span>
        </div>
        <div class="hw-meter-track">
            <div class="hw-meter-bar {status_fill}" style="width: {sim_percent:.1f}%;"></div>
            <div class="hw-needle" style="left: {thresh_percent:.1f}%;">
                <div class="hw-needle-label">τ {threshold:.2f}</div>
            </div>
        </div>
    </div>
    """
    render_html(html)


def render_similarity_meter(
    similarity: float,
    threshold: float,
    accepted: bool,
):
    """Backwards-compatible alias for render_similarity_gauge."""
    render_similarity_gauge(similarity, threshold, accepted)


def render_decision_panel(
    face_idx: int,
    identity: str,
    similarity: float,
    threshold: float,
    accepted: bool,
    detection_confidence: float,
):
    """Render physical biometric decision panel with LED status and explicit unknown rejection trace."""
    margin = similarity - threshold
    margin_str = f"+{margin:.3f}" if margin >= 0 else f"{margin:.3f}"
    status_class = "accepted" if accepted else "rejected"
    led_color = "green" if accepted else "red"
    status_text = "ACCESS ACCEPTED" if accepted else "ACCESS REJECTED (UNKNOWN)"

    display_identity = identity if accepted else "UNAUTHENTICATED SUBJECT"

    html = f"""
    <div class="decision-panel {status_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; border-bottom: 1px solid #162030; padding-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led {led_color}"></span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {'#10b981' if accepted else '#ef4444'}; letter-spacing: 0.05em;">
                    {status_text}
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748b;">
                CHANNEL #{face_idx + 1:02d}
            </span>
        </div>
        <div style="margin-bottom: 0.5rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: #64748b; text-transform: uppercase;">
                IDENTIFIED SUBJECT
            </div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; font-family: 'Inter', sans-serif;">
                {display_identity}
            </div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-top: 0.5rem;">
            <div>DETECTION CONF: <strong style="color: #f1f5f9;">{detection_confidence:.2f}</strong></div>
            <div>THRESHOLD τ: <strong style="color: #f59e0b;">{threshold:.2f}</strong></div>
        </div>
    </div>
    """
    render_html(html)


def render_analysis_trace(
    detected: bool,
    aligned: bool,
    embedded: bool,
    best_similarity: float,
    threshold: float,
    accepted: bool,
):
    """Render visual 6-stage analysis verification trace verifying unknown rejection."""
    decision_color = "#10b981" if accepted else "#ef4444"
    decision_text = "MATCH (ACCEPTED)" if accepted else "REJECTED (UNKNOWN)"

    html = f"""
    <div class="trace-box">
        <div style="font-weight: 700; color: #cbd5e1; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.4rem;">
            {ICONS['terminal']} BIOMETRIC VERIFICATION TRACE
        </div>
        <div class="trace-item">
            <span>[01] OPTICAL FACE ACQUISITION</span>
            <span style="color: #10b981;">{'✓ ACQUIRED' if detected else '✗ FAILED'}</span>
        </div>
        <div class="trace-item">
            <span>[02] 5-POINT AFFINE ALIGNMENT</span>
            <span style="color: #10b981;">{'✓ NORMALIZED' if aligned else '✗ FAILED'}</span>
        </div>
        <div class="trace-item">
            <span>[03] 128-D EMBEDDING EXTRACTION</span>
            <span style="color: #10b981;">{'✓ EXTRACTED' if embedded else '… WAITING'}</span>
        </div>
        <div class="trace-item">
            <span>[04] GALLERY COSINE SEARCH</span>
            <span style="color: #cbd5e1;">SIM: {best_similarity:.4f}</span>
        </div>
        <div class="trace-item">
            <span>[05] THRESHOLD BOUNDARY CHECK</span>
            <span style="color: #f59e0b;">τ = {threshold:.2f}</span>
        </div>
        <div class="trace-item" style="font-weight: 700;">
            <span>[06] HARDWARE DECISION</span>
            <span style="color: {decision_color};">{decision_text}</span>
        </div>
    </div>
    """
    render_html(html)


def render_hardware_stepper(steps: List[str], current_index: int):
    """Render 5-step industrial sequence stepper with physical LED indicators."""
    items_html = []
    for idx, name in enumerate(steps):
        is_done = idx < current_index
        is_curr = idx == current_index
        led_type = "green" if is_done else ("amber" if is_curr else "off")
        status_color = "#10b981" if is_done else ("#f59e0b" if is_curr else "#475569")

        items_html.append(
            f"""
            <div style="display: flex; align-items: center; gap: 0.45rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {status_color};">
                <span class="hw-led {led_type}"></span>
                <span>[{idx + 1:02d}] {name.upper()}</span>
            </div>
            """
        )
    joined = "".join(items_html)
    render_html(
        f"""
        <div style="display: flex; justify-content: space-between; background: #080b11; border: 1px solid #1c2536; border-radius: 6px; padding: 0.75rem 1.25rem; margin-bottom: 1.5rem; box-shadow: inset 0 2px 6px rgba(0,0,0,0.8);">
            {joined}
        </div>
        """
    )


def render_step_indicator(steps: List[str], current_index: int):
    """Backwards-compatible alias for render_hardware_stepper."""
    render_hardware_stepper(steps, current_index)


def render_hardware_status_bar(
    detector_name: str,
    embedder_name: str,
    device_info: str,
    threshold: float,
    identities_count: int,
    embeddings_count: int,
):
    """Render permanent bottom console status strip with chassis fasteners and telemetry."""
    html = f"""
    <div class="console-status-bar">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div class="hw-screw"></div>
            <span><span class="hw-led green"></span> SYSTEM OPERATIONAL</span>
            <span style="color: #475569;">|</span>
            <span>DETECTOR: <strong style="color: #cbd5e1;">{detector_name.upper()}</strong></span>
            <span style="color: #475569;">|</span>
            <span>EMBEDDER: <strong style="color: #cbd5e1;">{embedder_name.upper()}</strong></span>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span>DATABASE: <strong style="color: #10b981;">CONNECTED</strong> ({identities_count} ID / {embeddings_count} EMB)</span>
            <span style="color: #475569;">|</span>
            <span>TARGET: <strong style="color: #38bdf8;">{device_info.upper()}</strong></span>
            <span style="color: #475569;">|</span>
            <span>THRESHOLD: <strong style="color: #f59e0b;">τ {threshold:.2f}</strong></span>
            <div class="hw-screw"></div>
        </div>
    </div>
    """
    render_html(html)


def render_bios_boot_banner():
    """Render hardware BIOS self-test readout."""
    html = f"""
    <div style="background: #05070a; border: 1px solid #1a2232; border-radius: 4px; padding: 0.6rem 1rem; margin-bottom: 1.25rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748b; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="color: #f59e0b; font-weight: 700;">FACIAL RECOGNITION BIOS v2.4</span> // 
            POST: DETECTOR [<span style="color: #10b981;">OK</span>] 
            EMBEDDER [<span style="color: #10b981;">OK</span>] 
            SQLITE-WAL [<span style="color: #10b981;">OK</span>]
        </div>
        <div style="color: #10b981; font-weight: 700;">
            ● HARDWARE ONLINE
        </div>
    </div>
    """
    render_html(html)


def render_avatar(name: str) -> str:
    """Generate industrial personnel badge avatar initials."""
    parts = name.strip().split()
    initials = ""
    if len(parts) >= 2:
        initials = (parts[0][0] + parts[1][0]).upper()
    elif len(parts) == 1 and len(parts[0]) > 0:
        initials = parts[0][:2].upper()
    else:
        initials = "??"

    return f'<div style="width:36px; height:36px; border-radius:4px; background:linear-gradient(180deg, #242c3b 0%, #141822 100%); color:#f59e0b; border:1px solid #3b475a; box-shadow:inset 0 1px 0 rgba(255,255,255,0.1), 0 2px 4px rgba(0,0,0,0.6); display:flex; align-items:center; justify-content:center; font-weight:700; font-family:\'JetBrains Mono\', monospace; font-size:0.8125rem;">{initials}</div>'


def render_empty_state(
    title: str,
    description: str,
    icon_name: str = "search",
):
    """Render recessed optical radar / sensor inactive empty state."""
    icon_svg = ICONS.get(icon_name, "")
    html = f"""
    <div style="background: #080b11; border: 1px dashed #242d3e; border-radius: 8px; padding: 2.5rem 2rem; text-align: center; max-width: 500px; margin: 2rem auto; box-shadow: inset 0 3px 12px rgba(0,0,0,0.85);">
        <div style="color: #f59e0b; margin-bottom: 0.75rem; display: flex; justify-content: center;">
            {icon_svg}
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
            {title}
        </div>
        <div style="font-size: 0.8125rem; color: #64748b; line-height: 1.5;">
            {description}
        </div>
    </div>
    """
    render_html(html)


def bgr_to_base64_img(img_bgr: np.ndarray) -> str:
    """Convert OpenCV BGR image into base64 data URI for inline HTML rendering."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    success, buffer = cv2.imencode(".jpg", img_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def render_workflow_mode_badge(mode: str = "identification") -> None:
    """Render high-visibility workflow banner stating whether biometric profiles will be saved or transiently queried."""
    is_enroll = mode.lower() in ("enrollment", "registration", "enroll")
    if is_enroll:
        border_color = "#10b981"
        glow_color = "rgba(16, 185, 129, 0.15)"
        text_color = "#10b981"
        led = "green"
        mode_tag = "REGISTRATION MODE"
        desc = "Captured reference will be added to the database."
        sub = "BIOMETRIC ENROLLMENT ACTIVE"
    else:
        border_color = "#f59e0b"
        glow_color = "rgba(245, 158, 11, 0.15)"
        text_color = "#f59e0b"
        led = "amber"
        mode_tag = "QUERY MODE"
        desc = "No biometric profile will be added."
        sub = "TRANSIENT RECOGNITION AUDIT"

    html = f"""
    <div style="background: #080b11; border: 1px solid {border_color}; border-left: 5px solid {border_color}; border-radius: 6px; padding: 0.65rem 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 0 15px {glow_color}; font-family: 'JetBrains Mono', monospace;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span class="hw-led {led}"></span>
            <div>
                <span style="font-size: 0.8125rem; font-weight: 800; color: {text_color}; letter-spacing: 0.05em;">{mode_tag}:</span>
                <span style="font-size: 0.8125rem; color: #f8fafc; margin-left: 0.4rem;">{desc}</span>
            </div>
        </div>
        <span style="font-size: 0.6875rem; color: #64748b; font-weight: 700; letter-spacing: 0.08em;">{sub}</span>
    </div>
    """
    render_html(html)


def render_capture_source_selector(
    active_source: str = "upload",
    workflow_title: str = "ACQUISITION SOURCE SELECTOR",
) -> None:
    """Render physical console optical source rocker switch plate."""
    is_upload = active_source == "upload"
    is_webcam = active_source == "webcam"

    upload_led = "green" if is_upload else "off"
    webcam_led = "amber" if is_webcam else "off"

    html = f"""
    <div class="source-rocker-bay">
        <div class="source-rocker-title">
            <span>{workflow_title.upper()}</span>
            <span>CH: OPTICAL-INPUT</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div style="background: {'linear-gradient(180deg, #101520 0%, #0a0d14 100%)' if is_upload else 'linear-gradient(180deg, #1f2737 0%, #151a24 100%)'}; border: 1px solid {'#10b981' if is_upload else '#283345'}; border-radius: 6px; padding: 0.5rem 0.75rem; display: flex; align-items: center; justify-content: space-between; box-shadow: {'inset 0 2px 6px rgba(0,0,0,0.85), 0 0 10px rgba(16, 185, 129, 0.2)' if is_upload else '0 2px 4px rgba(0,0,0,0.4)'};">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {'#10b981' if is_upload else '#94a3b8'};">
                    [01] IMAGE UPLOAD
                </span>
                <span class="hw-led {upload_led}"></span>
            </div>
            <div style="background: {'linear-gradient(180deg, #101520 0%, #0a0d14 100%)' if is_webcam else 'linear-gradient(180deg, #1f2737 0%, #151a24 100%)'}; border: 1px solid {'#f59e0b' if is_webcam else '#283345'}; border-radius: 6px; padding: 0.5rem 0.75rem; display: flex; align-items: center; justify-content: space-between; box-shadow: {'inset 0 2px 6px rgba(0,0,0,0.85), 0 0 10px rgba(245, 158, 11, 0.2)' if is_webcam else '0 2px 4px rgba(0,0,0,0.4)'};">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {'#f59e0b' if is_webcam else '#94a3b8'};">
                    [02] WEBCAM CONSOLE
                </span>
                <span class="hw-led {webcam_led}"></span>
            </div>
        </div>
    </div>
    """
    render_html(html)


def render_enrollment_source_selector(active_source: str) -> None:
    """Render physical console enrollment source rocker switch plate."""
    render_capture_source_selector(active_source, "ENROLLMENT SOURCE SELECTOR")


def render_webcam_monitor_bezel(
    channel_name: str = "PRIMARY OPTICAL FEED",
    status_text: str = "LIVE",
    led_color: str = "green",
    device_label: str = "DEFAULT WEBCAM",
) -> None:
    """Render physical camera monitor bezel header with status indicator and telemetry tags."""
    html = f"""
    <div class="webcam-telemetry-header">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div class="hw-screw"></div>
            <span>CAMERA INPUT</span>
            <span class="hw-led {led_color}"></span>
            <strong style="color: {'#10b981' if led_color == 'green' else '#f59e0b'};">{status_text.upper()}</strong>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #64748b;">FEED:</span>
            <span style="color: #cbd5e1;">{channel_name}</span>
            <span style="color: #475569;">|</span>
            <span style="color: #64748b;">DEV:</span>
            <span style="color: #f59e0b;">{device_label}</span>
            <div class="hw-screw"></div>
        </div>
    </div>
    """
    render_html(html)


def render_face_alignment_guide() -> None:
    """Render biometric optical reticle alignment guidelines."""
    html = f"""
    <div class="camera-guidance-pill valid" style="margin-bottom: 0.5rem;">
        <span class="hw-led amber"></span>
        <span>OPTICAL RETICLE: Position frontal face within center viewport (yaw &amp; pitch &lt; 30°)</span>
    </div>
    """
    render_html(html)


def render_camera_permission_notice(is_denied: bool = False) -> None:
    """Render camera permission guidance or denied warning plate."""
    if is_denied:
        html = f"""
        <div class="panel-plate" style="border-left: 4px solid #ef4444; margin-bottom: 1rem; padding: 1rem 1.25rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #ef4444; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led red"></span> CAMERA ACCESS DENIED OR UNAVAILABLE
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.5; font-family: 'Inter', sans-serif;">
                Browser camera permission is required for live optical capture. Check your browser permissions bar or switch to <strong>Image Upload</strong> mode to enroll from local photo files.
            </div>
        </div>
        """
    else:
        html = f"""
        <div style="background: #080b11; border: 1px solid #1c2536; border-radius: 6px; padding: 0.6rem 0.85rem; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #94a3b8;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led blue"></span>
                <span>CAMERA ACCESS: Browser permission engaged for live biometric capture</span>
            </div>
            <span style="color: #10b981; font-weight: 700;">● PRIVACY ENFORCED</span>
        </div>
        """
    render_html(html)


def render_captured_frame_review(
    num_faces: int,
    is_valid_single: bool,
    quality_score: Optional[float] = None,
    laplacian_var: Optional[float] = None,
    resolution_str: Optional[str] = None,
) -> None:
    """Render captured frame review status panel."""
    if num_faces == 1 and is_valid_single:
        status_color = "#10b981"
        led_color = "green"
        headline = "SINGLE SUBJECT ACQUIRED — READY FOR BIOMETRIC ENROLLMENT"
    elif num_faces == 0:
        status_color = "#f59e0b"
        led_color = "amber"
        headline = "NO FACE DETECTED — RE-ALIGN SUBJECT AND RETAKE"
    else:
        status_color = "#ef4444"
        led_color = "red"
        headline = f"POLICY VIOLATION: {num_faces} FACES DETECTED (EXACTLY 1 REQUIRED)"

    qual_items = []
    if resolution_str:
        qual_items.append(f"RES: <strong style='color:#f8fafc;'>{resolution_str}</strong>")
    if laplacian_var is not None:
        qual_items.append(f"LAPLACIAN SHARPNESS: <strong style='color:{'#10b981' if laplacian_var >= 25.0 else '#ef4444'};'>{laplacian_var:.1f}</strong>")

    qual_html = " | ".join(qual_items) if qual_items else ""

    html = f"""
    <div style="background: #080b11; border: 1px solid #1c2536; border-left: 4px solid {status_color}; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.75rem; margin-bottom: 0.75rem;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {status_color}; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span class="hw-led {led_color}"></span> {headline}
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: #64748b;">
            {qual_html}
        </div>
    </div>
    """
    render_html(html)
