"""People directory view: Personnel registry rack, reference audit drawer, and industrial deletion."""

from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

from app.config import config
from app.database import DatabaseManager
from app.ui.components import (
    render_avatar,
    render_empty_state,
    render_hardware_header,
)
from app.ui.theme import ICONS, THEME, render_html


def render_people_view(db_mgr: DatabaseManager):
    """Render the physical personnel registry and mechanical inspection drawer."""
    render_hardware_header(
        page_title="PERSONNEL REGISTRY",
        subtitle="BIOMETRIC DOSSIER ARCHIVE & REFERENCE EMBEDDING AUDIT",
        model_status="DATABASE CONNECTED",
        device="SQLITE WAL",
    )

    persons = db_mgr.get_persons_with_counts()
    if not persons:
        render_empty_state(
            title="PERSONNEL ARCHIVE EMPTY",
            description="No biometric profiles enrolled. Initiate registration to populate the hardware gallery.",
            icon_name="people",
        )
        if st.button("➕ INITIATE FIRST REGISTRATION", type="primary"):
            st.session_state["nav"] = "enroll"
            st.rerun()
        return

    # Search & Registry Stats Bar
    search_col, stat_col = st.columns([3, 1], vertical_alignment="center")
    with search_col:
        search_query = st.text_input(
            "Filter Registry",
            placeholder="Search by subject designation...",
            label_visibility="collapsed",
        )
    with stat_col:
        render_html(
            f"""
            <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8;">
                ARCHIVE TOTAL: <strong style="color: {THEME['accent_amber']};">{len(persons):03d}</strong> SUBJECTS
            </div>
            """
        )

    filtered_persons = [
        p for p in persons if search_query.lower() in p["name"].lower()
    ]

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Two-Column Layout: Subject Rack on Left, Mechanical Inspection Drawer on Right
    col_rack, col_drawer = st.columns([3, 2])

    with col_rack:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led green"></span> ENROLLED SUBJECT RACK
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                SELECT SUBJECT TO ENGAGE DOSSIER INSPECTOR
            </div>
            """
        )

        if not filtered_persons:
            st.info("No matching personnel records in archive.")
        else:
            selected_id = st.session_state.get("selected_person_id")
            if selected_id is None and filtered_persons:
                selected_id = filtered_persons[0]["id"]
                st.session_state["selected_person_id"] = selected_id

            for p in filtered_persons:
                p_id = p["id"]
                p_name = p["name"]
                emb_count = p["embedding_count"]
                created = p["created_at"]
                is_selected = (selected_id == p_id)

                row_av, row_info, row_btn = st.columns([0.6, 4.4, 2.0], vertical_alignment="center")
                with row_av:
                    render_html(render_avatar(p_name))
                with row_info:
                    render_html(
                        f"""
                        <div style="font-weight: 700; font-size: 0.9375rem; color: #f8fafc; line-height: 1.2;">{p_name}</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748b; margin-top: 0.2rem;">
                            ID #{p_id:03d} &bull; {emb_count:02d} vector(s) &bull; Enrolled {created[:10] if created else 'N/A'}
                        </div>
                        """
                    )
                with row_btn:
                    btn_type = "primary" if is_selected else "secondary"
                    btn_label = "● ACTIVE" if is_selected else "INSPECT"
                    if st.button(btn_label, key=f"sel_{p_id}", type=btn_type, use_container_width=True):
                        st.session_state["selected_person_id"] = p_id
                        st.rerun()

                st.markdown("<div style='border-bottom: 1px solid #1c2536; margin: 0.5rem 0 0.75rem 0;'></div>", unsafe_allow_html=True)

    # Mechanical Inspection Drawer
    with col_drawer:
        render_html(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
                <span class="hw-led blue"></span> MECHANICAL INSPECTION DRAWER
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                BIOMETRIC DOSSIER & CASING AUDIT
            </div>
            """
        )

        selected_id = st.session_state.get("selected_person_id")
        selected_person = next((p for p in persons if p["id"] == selected_id), None)

        if selected_person is None and persons:
            selected_person = persons[0]
            st.session_state["selected_person_id"] = selected_person["id"]

        if selected_person:
            p_id = selected_person["id"]
            p_name = selected_person["name"]
            emb_count = selected_person["embedding_count"]
            created = selected_person["created_at"]

            embeddings = db_mgr.get_embeddings_for_person(p_id)
            cam_count = sum(1 for e in embeddings if e.get("source") == "webcam")
            up_count = sum(1 for e in embeddings if e.get("source") != "webcam")

            render_html(
                f"""
                <div class="panel-plate" style="margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.75rem;">
                        {render_avatar(p_name)}
                        <div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: 'Inter', sans-serif;">
                                {p_name}
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {THEME['accent_amber']};">
                                SUBJECT SERIAL #{p_id:04d}
                            </div>
                        </div>
                    </div>
                    <div class="glass-display" style="padding: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #94a3b8; line-height: 1.8;">
                        <div>• ENROLLED: <strong style="color: #f1f5f9;">{created}</strong></div>
                        <div>• VECTORS: <strong style="color: #f59e0b;">{emb_count:02d} REFERENCE EMBEDDINGS</strong> (Webcam: {cam_count} | Upload: {up_count})</div>
                        <div>• DIMENSION: <strong style="color: #10b981;">128-D L2 NORMALIZED</strong></div>
                        <div>• POLICY: <strong style="color: #38bdf8;">MAX COSINE SIMILARITY</strong></div>
                    </div>
                </div>
                """
            )

            # Audit Reference Embeddings & Photos
            render_html(
                """
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #94a3b8; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                    <span>REFERENCE EMBEDDINGS AUDIT</span>
                    <span class="hw-led green"></span>
                </div>
                """
            )

            if embeddings:
                emb_rows = []
                for e in embeddings:
                    src_tag = "WEBCAM" if e.get("source") == "webcam" else "UPLOAD"
                    src_color = THEME["accent_amber"] if src_tag == "WEBCAM" else THEME["status_green"]
                    img_label = e.get("source_image_name") or f"vec_{e['id']}"
                    emb_rows.append(
                        f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; background:#080b11; border:1px solid #1c2436; border-radius:4px; padding:0.35rem 0.6rem; margin-bottom:0.3rem; font-family:'JetBrains Mono', monospace; font-size:0.6875rem;">
                            <span style="color:#cbd5e1;">#{e['id']:02d} &bull; {img_label}</span>
                            <span style="color:{src_color}; font-weight:700;">[{src_tag}]</span>
                        </div>
                        """
                    )
                render_html("".join(emb_rows))

            ref_folder = config.enrolled_dir / p_name
            ref_images = list(ref_folder.glob("*.*")) if ref_folder.exists() else []

            if ref_images:
                st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
                num_to_show = min(3, len(ref_images))
                img_cols = st.columns(num_to_show)
                for idx in range(num_to_show):
                    img_path = ref_images[idx]
                    with img_cols[idx]:
                        try:
                            im = Image.open(img_path)
                            st.image(im, use_container_width=True, caption=img_path.name[:16])
                        except Exception:
                            pass
            else:
                st.caption("No disk reference image files stored for this subject.")

            # Deletion with Industrial Safety Confirmation Box
            st.markdown("<div style='border-top: 1px solid #1e2638; margin: 1.25rem 0;'></div>", unsafe_allow_html=True)
            render_html(
                """
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #ef4444; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.4rem;">
                    {ICONS['alert']} DANGER ZONE: ARCHIVE PURGE
                </div>
                """
            )

            if st.checkbox(f"Engage safety override to purge '{p_name}'", key=f"confirm_check_{p_id}"):
                st.warning(
                    f"⚠ SAFETY OVERRIDE ENGAGED: Purging '{p_name}' will permanently delete this identity and cascade erase all {emb_count} reference vectors."
                )
                if st.button(f"🚨 CONFIRM PERMANENT PURGE", type="primary", key=f"del_btn_{p_id}", use_container_width=True):
                    success = db_mgr.delete_person(p_id)
                    if success:
                        st.session_state["selected_person_id"] = None
                        st.success(f"Subject '{p_name}' and all associated reference embeddings purged.")
                        st.rerun()
                    else:
                        st.error("Failed to purge record from SQLite.")
