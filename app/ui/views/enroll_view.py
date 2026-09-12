"""Enrollment view: Personnel registration console, dual-source acquisition (Upload/Webcam), and biometric validation."""

from typing import Optional
import cv2
import numpy as np
import streamlit as st

from app.config import config
from app.database import DatabaseManager
from app.enrollment.service import EnrollmentService
from app.ui.components import (
    render_camera_permission_notice,
    render_captured_frame_review,
    render_enrollment_source_selector,
    render_face_alignment_guide,
    render_hardware_header,
    render_hardware_stepper,
    render_webcam_monitor_bezel,
)
from app.ui.theme import ICONS, THEME, render_html
from app.utils.image_utils import (
    bgr_to_rgb,
    crop_face,
    draw_detection_annotations,
    load_image,
)


def render_enroll_view(
    enroll_service: EnrollmentService,
    db_mgr: DatabaseManager,
):
    """Render the physical enrollment workstation with Image Upload and Webcam acquisition modes."""
    render_hardware_header(
        page_title="PERSONNEL REGISTRATION",
        subtitle="BIOMETRIC PROFILE ENROLLMENT, WEBCAM VIEWPORT & QUALITY AUDIT WORKBENCH",
        model_status="READY",
        device="CPU",
    )

    # Initialize session state tracking
    if "enroll_source_mode" not in st.session_state:
        st.session_state["enroll_source_mode"] = "upload"
    if "enroll_subject_name" not in st.session_state:
        st.session_state["enroll_subject_name"] = ""
    if "cam_session_ver" not in st.session_state:
        st.session_state["cam_session_ver"] = 0

    active_source = st.session_state["enroll_source_mode"]

    # 4-Stage Hardware Stepper
    step_names = [
        "SUBJECT DOSSIER",
        "SOURCE SELECTOR",
        "OPTICAL ACQUISITION",
        "DATABASE COMMIT",
    ]
    current_step = 0

    # =========================================================================
    # 1. SUBJECT INFORMATION & DUAL-SOURCE ROCKER SELECTOR
    # =========================================================================
    col_sub_info, col_src_rocker = st.columns([1, 1])

    with col_sub_info:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led amber"></span> SUBJECT IDENTIFIER
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem; font-family: 'JetBrains Mono', monospace;">
                SPECIFY SUBJECT FULL NAME OR UNIQUE OPERATOR ID
            </div>
            """
        )
        name_input = st.text_input(
            "Subject Name",
            value=st.session_state["enroll_subject_name"],
            placeholder="e.g. Alice Smith (ID: AS-001)",
            label_visibility="collapsed",
            help="Alphanumeric identifier for the subject being enrolled.",
        )
        st.session_state["enroll_subject_name"] = name_input
        if name_input.strip():
            current_step = 1

    with col_src_rocker:
        # Hardware Source Selector Visual Plate
        render_enrollment_source_selector(active_source)

        # Tactile Mode Selection Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(
                "📁 [01] IMAGE UPLOAD",
                type="primary" if active_source == "upload" else "secondary",
                use_container_width=True,
                key="btn_switch_upload",
            ):
                st.session_state["enroll_source_mode"] = "upload"
                st.rerun()

        with col_btn2:
            if st.button(
                "📷 [02] WEBCAM CAPTURE",
                type="primary" if active_source == "webcam" else "secondary",
                use_container_width=True,
                key="btn_switch_webcam",
            ):
                st.session_state["enroll_source_mode"] = "webcam"
                st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 2. WORKSPACE LAYOUT: Acquisition on Left, Sensors & Dossier on Right
    # =========================================================================
    col_workspace, col_telemetry = st.columns([3, 2])

    # -------------------------------------------------------------------------
    # WORKSPACE: IMAGE UPLOAD MODE
    # -------------------------------------------------------------------------
    if active_source == "upload":
        with col_workspace:
            render_html(
                """
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span class="hw-led green"></span> OPTICAL FRAME INGESTION (FILE UPLOAD)
                </div>
                """
            )
            enroll_file = st.file_uploader(
                "Reference Optical Frame",
                type=["jpg", "jpeg", "png", "webp"],
                key="enroll_file_uploader",
                help="Upload a high-resolution frontal portrait containing exactly 1 subject.",
            )

            skip_qc = st.checkbox(
                "Bypass biometric sharpness & resolution gates (Manual Override)",
                value=False,
                key="upload_skip_qc",
                help="Bypass Laplacian blur variance and minimum bounding box checks.",
            )

            if enroll_file is not None:
                current_step = 2
                try:
                    img_bytes = enroll_file.getvalue()
                    raw_bgr = load_image(img_bytes)
                    dets = enroll_service.detector.detect(raw_bgr)

                    render_html(
                        f"""
                        <div class="camera-monitor-bezel">
                            <div class="camera-corner-tag">
                                <span>[ OPTICAL INGESTION PREVIEW ]</span>
                                <span>FACES DETECTED: {len(dets)}</span>
                            </div>
                        </div>
                        """
                    )

                    if len(dets) == 1:
                        current_step = 3
                        target = dets[0]
                        annotated = draw_detection_annotations(raw_bgr, dets)
                        st.image(
                            bgr_to_rgb(annotated),
                            caption="Single face localized with 5 facial landmark anchors",
                            use_container_width=True,
                        )
                        face_crop = crop_face(raw_bgr, target.bbox, margin=0.1)
                        lap_var = float(
                            cv2.Laplacian(face_crop, cv2.CV_64F).var()
                        )
                        res_str = f"{face_crop.shape[1]}x{face_crop.shape[0]} px"
                        render_captured_frame_review(
                            num_faces=1,
                            is_valid_single=True,
                            laplacian_var=lap_var,
                            resolution_str=res_str,
                        )
                    elif len(dets) == 0:
                        st.image(bgr_to_rgb(raw_bgr), use_container_width=True)
                        render_captured_frame_review(
                            num_faces=0, is_valid_single=False
                        )
                    else:
                        annotated = draw_detection_annotations(raw_bgr, dets)
                        st.image(bgr_to_rgb(annotated), use_container_width=True)
                        render_captured_frame_review(
                            num_faces=len(dets), is_valid_single=False
                        )

                except Exception as e:
                    st.error(f"Optical frame decoding error: {e}")

            st.markdown(
                "<div style='height: 0.5rem;'></div>", unsafe_allow_html=True
            )
            upload_submit = st.button(
                "➕ COMMIT UPLOAD TO DATABASE",
                type="primary",
                use_container_width=True,
            )

            if upload_submit:
                if not name_input.strip():
                    st.error("Please specify a subject identifier before committing.")
                elif enroll_file is None:
                    st.error("Please provide a reference image file.")
                else:
                    with st.spinner("Executing localization, alignment, embedding extraction, and persistence..."):
                        res = enroll_service.enroll(
                            name=name_input.strip(),
                            image=enroll_file.getvalue(),
                            allow_save_image=True,
                            skip_quality_check=skip_qc,
                            source_type="upload",
                        )
                    _handle_enrollment_result(res)

    # -------------------------------------------------------------------------
    # WORKSPACE: WEBCAM LIVE CAPTURE MODE
    # -------------------------------------------------------------------------
    else:
        with col_workspace:
            # Privacy & Permission Badge
            render_camera_permission_notice(is_denied=False)

            # Biometric Webcam Monitor Bezel & Guides
            render_webcam_monitor_bezel(
                channel_name="CH-01 // WEBCAM STREAM",
                status_text="LIVE",
                led_color="green",
                device_label="INTEGRATED OPTICAL SENSOR",
            )
            render_face_alignment_guide()

            # Streamlit First-Class WebRTC Camera Widget
            cam_key = f"cam_widget_{st.session_state['cam_session_ver']}"
            cam_picture = st.camera_input(
                "Live Biometric Camera Input",
                key=cam_key,
                label_visibility="collapsed",
                help="Align face in center frame and click 'Take Photo'.",
            )

            skip_qc_cam = st.checkbox(
                "Bypass biometric sharpness & resolution gates (Manual Override)",
                value=False,
                key="cam_skip_qc",
                help="Bypass Laplacian blur variance and minimum bounding box checks.",
            )

            # Frame Capture & Instant Biometric Review
            if cam_picture is not None:
                current_step = 2
                try:
                    cam_bytes = cam_picture.getvalue()
                    raw_bgr = load_image(cam_bytes)
                    dets = enroll_service.detector.detect(raw_bgr)

                    render_html(
                        f"""
                        <div class="panel-plate" style="margin-top: 1rem; border-top: 2px solid {THEME['accent_amber']};">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #f8fafc;">
                                    {ICONS['terminal']} CAPTURED FRAME REVIEW
                                </span>
                                <span class="hw-badge active-green">FACES: {len(dets)}</span>
                            </div>
                        </div>
                        """
                    )

                    if len(dets) == 1:
                        current_step = 3
                        target = dets[0]
                        annotated = draw_detection_annotations(raw_bgr, dets)
                        st.image(
                            bgr_to_rgb(annotated),
                            caption="Live Frame Frozen — Single subject verified with 5 facial landmarks",
                            use_container_width=True,
                        )
                        face_crop = crop_face(raw_bgr, target.bbox, margin=0.1)
                        lap_var = float(
                            cv2.Laplacian(face_crop, cv2.CV_64F).var()
                        )
                        res_str = f"{face_crop.shape[1]}x{face_crop.shape[0]} px"
                        render_captured_frame_review(
                            num_faces=1,
                            is_valid_single=True,
                            laplacian_var=lap_var,
                            resolution_str=res_str,
                        )

                        cam_commit = st.button(
                            "➕ COMMIT WEBCAM REFERENCE TO DATABASE",
                            type="primary",
                            use_container_width=True,
                        )
                        if cam_commit:
                            if not name_input.strip():
                                st.error("Please specify a subject identifier before committing.")
                            else:
                                with st.spinner("Processing biometric embedding and updating database..."):
                                    res = enroll_service.enroll(
                                        name=name_input.strip(),
                                        image=cam_bytes,
                                        allow_save_image=True,
                                        skip_quality_check=skip_qc_cam,
                                        source_type="webcam",
                                    )
                                _handle_enrollment_result(res)

                    elif len(dets) == 0:
                        st.image(bgr_to_rgb(raw_bgr), use_container_width=True)
                        render_captured_frame_review(
                            num_faces=0, is_valid_single=False
                        )
                        st.warning(
                            "⚠ NO FACE LOCALIZED: Position your face inside the camera viewfinder and click 'Clear photo' on the camera display to retake."
                        )
                    else:
                        annotated = draw_detection_annotations(raw_bgr, dets)
                        st.image(bgr_to_rgb(annotated), use_container_width=True)
                        render_captured_frame_review(
                            num_faces=len(dets), is_valid_single=False
                        )
                        st.error(
                            f"⚠ SINGLE-FACE POLICY VIOLATION: {len(dets)} faces localized. Enrollment requires exactly 1 subject. Clear extra individuals from view and click 'Clear photo' to retake."
                        )

                except Exception as e:
                    st.error(f"Webcam frame analysis error: {e}")

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: HARDWARE QUALITY SENSORS & IDENTITY DOSSIER AUDIT
    # -------------------------------------------------------------------------
    with col_telemetry:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led blue"></span> BIOMETRIC INGESTION GATES
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                AUTOMATED PRE-INGESTION HARDWARE CRITERIA
            </div>
            """
        )

        render_html(
            f"""
            <div class="glass-display" style="margin-bottom: 1.25rem; padding: 1rem 1.25rem;">
                <div style="display: flex; flex-direction: column; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> SINGLE SUBJECT: Exactly 1 face per reference frame
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> RESOLUTION GATE: Face bounding box &ge; {config.min_face_size}x{config.min_face_size} px
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> SHARPNESS GATE: Laplacian variance &ge; {config.min_laplacian_variance:.1f}
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> POSE ORIENTATION: Frontal yaw &amp; pitch &lt; 45°
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led amber"></span> DUAL SOURCE: Image upload or live WebRTC camera
                    </div>
                </div>
            </div>
            """
        )

        # Subject Dossier Audit if name matches existing records
        if name_input.strip():
            existing = db_mgr.get_person_by_name(name_input.strip())
            if existing:
                embeddings = db_mgr.get_embeddings_for_person(existing["id"])
                cam_count = sum(1 for e in embeddings if e.get("source") == "webcam")
                up_count = sum(1 for e in embeddings if e.get("source") != "webcam")

                emb_pills = []
                for e in embeddings[-4:]:
                    src_tag = "WEBCAM" if e.get("source") == "webcam" else "UPLOAD"
                    src_color = THEME["accent_amber"] if src_tag == "WEBCAM" else THEME["status_green"]
                    emb_pills.append(
                        f"""
                        <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #94a3b8; padding: 0.2rem 0; border-bottom: 1px solid #141b27;">
                            <span>EMB #{e['id']} ({e.get('source_image_name', 'ref')})</span>
                            <strong style="color: {src_color};">[{src_tag}]</strong>
                        </div>
                        """
                    )
                pills_html = "".join(emb_pills)

                render_html(
                    f"""
                    <div class="panel-plate" style="margin-bottom: 1rem;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #cbd5e1; margin-bottom: 0.35rem; display: flex; justify-content: space-between;">
                            <span>KNOWN IDENTITY DOSSIER</span>
                            <span class="hw-led green"></span>
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem;">
                            {existing['name']}
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem;">
                            TOTAL EMBEDDINGS: <strong style="color:{THEME['accent_amber']};">{len(embeddings)}</strong> 
                            (Webcam: {cam_count} | Upload: {up_count})
                        </div>
                        <div style="border-top: 1px solid #1a2232; padding-top: 0.4rem;">
                            {pills_html}
                        </div>
                    </div>
                    """
                )

    # Sequence Stepper
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    render_hardware_stepper(step_names, current_step)


def _handle_enrollment_result(result):
    """Render biometric enrollment confirmation plate and multi-reference controls."""
    if result.success:
        st.session_state["last_enrolled_name"] = result.person_name
        src_label = result.source_type.upper()
        src_color = THEME["accent_amber"] if src_label == "WEBCAM" else THEME["status_green"]

        render_html(
            f"""
            <div class="panel-plate" style="border-left: 4px solid #10b981; padding: 1.25rem 1.5rem; margin-top: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 0.5rem;">
                        <span class="hw-led green"></span> BIOMETRIC ENROLLMENT CONFIRMED
                    </span>
                    <span class="hw-badge active-green">SOURCE: {src_label}</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; color: #94a3b8; line-height: 1.5;">
                    SUBJECT: <strong style="color: #f1f5f9;">{result.person_name}</strong> | 
                    STORED REFERENCE VECTORS: <strong style="color: {THEME['accent_amber']};">{result.total_embeddings_for_person}</strong> |
                    INGESTION: <strong style="color: {src_color};">{src_label}</strong>
                </div>
            </div>
            """
        )

        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button(
                "📷 + CAPTURE ANOTHER REFERENCE",
                type="primary",
                use_container_width=True,
                key="btn_multi_ref_another",
                help="Capture another reference image for this identity from another angle or lighting.",
            ):
                st.session_state["cam_session_ver"] = st.session_state.get("cam_session_ver", 0) + 1
                st.rerun()

        with col_act2:
            if st.button(
                "⚡ ENGAGE IDENTIFICATION TEST",
                use_container_width=True,
                key="btn_goto_identify",
            ):
                st.session_state["nav"] = "identify"
                st.rerun()

        with col_act3:
            if st.button(
                "👥 AUDIT IN PEOPLE DIRECTORY",
                use_container_width=True,
                key="btn_goto_people",
            ):
                st.session_state["nav"] = "people"
                st.rerun()

    else:
        st.error(f"❌ ENROLLMENT REJECTED: {result.message}")
        if result.quality and result.quality.issues:
            st.warning("Quality check failures: " + "; ".join(result.quality.issues))
