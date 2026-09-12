"""Centralized skeuomorphic design system, CSS variables, and styling tokens for VISION-ID Console."""

# Hardware Palette & Physical Design Tokens
THEME = {
    # Chassis & Material Surfaces
    "chassis_bg": "#0c0e14",
    "chassis_gradient": "linear-gradient(180deg, #131720 0%, #0c0e14 100%)",
    "panel_bg": "#161b24",
    "panel_gradient": "linear-gradient(180deg, #1e2431 0%, #151a24 100%)",
    "panel_surface": "#1a202c",
    "panel_inset": "#0d1017",
    "metal_border": "#2c3545",
    "metal_border_subtle": "#1e2430",
    "metal_border_light": "rgba(255, 255, 255, 0.08)",
    "metal_border_dark": "rgba(0, 0, 0, 0.65)",
    
    # Inset Display Glass
    "glass_bg": "#06080d",
    "glass_gradient": "radial-gradient(ellipse at 50% 0%, #0e1420 0%, #05070a 100%)",
    "glass_border": "#151b27",
    
    # Typography
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "text_phosphor_amber": "#f59e0b",
    "text_phosphor_green": "#10b981",
    "text_phosphor_red": "#ef4444",
    "text_phosphor_blue": "#38bdf8",
    
    # Hardware Accent & Indicators
    "accent_amber": "#f59e0b",
    "accent_amber_dim": "#b45309",
    "accent_amber_glow": "rgba(245, 158, 11, 0.35)",
    
    # Operational Status
    "status_green": "#10b981",
    "status_green_glow": "rgba(16, 185, 129, 0.4)",
    "status_amber": "#f59e0b",
    "status_amber_glow": "rgba(245, 158, 11, 0.4)",
    "status_red": "#ef4444",
    "status_red_glow": "rgba(239, 68, 68, 0.4)",
    "status_blue": "#3b82f6",
    "status_blue_glow": "rgba(59, 130, 246, 0.4)",
    
    # Legacy alias support
    "bg": "#0c0e14",
    "surface": "#161b24",
    "surface_elevated": "#1e2431",
    "surface_hover": "#252d3d",
    "border": "#2c3545",
    "border_subtle": "#1e2430",
    "accent": "#f59e0b",
    "accent_hover": "#d97706",
    "accent_subtle": "rgba(245, 158, 11, 0.12)",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#38bdf8",
}

# Lucide-style SVG icons embedded for crisp instrument rendering
ICONS = {
    "dashboard": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/></svg>""",
    "identify": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="11" r="3"/><path d="M16 16.5a4 4 0 0 0-8 0"/></svg>""",
    "enroll": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" x2="19" y1="8" y2="14"/><line x1="22" x2="16" y1="11" y2="11"/></svg>""",
    "people": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>""",
    "evaluation": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>""",
    "settings": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>""",
    "check": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>""",
    "alert": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>""",
    "shield": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>""",
    "cpu": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>""",
    "database": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>""",
    "search": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/></svg>""",
    "trash": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>""",
    "upload": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>""",
    "terminal": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>""",
    "activity": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>""",
}


def get_global_css() -> str:
    """Return comprehensive stylesheet injecting the skeuomorphic physical console design language."""
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {{
            --chassis-bg: {THEME['chassis_bg']};
            --panel-bg: {THEME['panel_bg']};
            --panel-surface: {THEME['panel_surface']};
            --glass-bg: {THEME['glass_bg']};
            --metal-border: {THEME['metal_border']};
            --metal-border-subtle: {THEME['metal_border_subtle']};
            --text-primary: {THEME['text_primary']};
            --text-secondary: {THEME['text_secondary']};
            --text-muted: {THEME['text_muted']};
            --accent-amber: {THEME['accent_amber']};
            --accent-glow: {THEME['accent_amber_glow']};
            --status-green: {THEME['status_green']};
            --status-red: {THEME['status_red']};
            --status-blue: {THEME['status_blue']};
            
            /* Legacy mapping */
            --surface: {THEME['panel_bg']};
            --border: {THEME['metal_border']};
            --accent: {THEME['accent_amber']};
            --success: {THEME['status_green']};
            --danger: {THEME['status_red']};
        }}

        /* Reset & Chassis Canvas */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: var(--chassis-bg) !important;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                linear-gradient(180deg, #10141d 0%, #0a0c12 100%) !important;
            background-size: 32px 32px, 32px 32px, 100% 100% !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }}

        /* Hide Streamlit Default Chrome */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0px !important;
            display: none !important;
        }}
        footer {{ display: none !important; }}
        #MainMenu {{ visibility: hidden !important; }}
        .stDeployButton {{ display: none !important; }}

        /* Main Workspace Container */
        .block-container {{
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1440px !important;
        }}

        /* Industrial Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: #090c12;
            border-left: 1px solid #161c28;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #252e3e;
            border-radius: 3px;
            border: 1px solid #364257;
            box-shadow: inset 1px 1px 0 rgba(255,255,255,0.1);
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #334057;
        }}

        /* Physical Rack Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #141822 0%, #0d1017 100%) !important;
            border-right: 2px solid #232b3a !important;
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.6) !important;
        }}
        [data-testid="stSidebarContent"] {{
            padding-top: 1.25rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
        [data-testid="stSidebarNav"] {{ display: none !important; }}

        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em !important;
        }}
        .font-mono {{
            font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        }}

        /* =========================================================================
           SKEUOMORPHIC HARDWARE COMPONENTS
           ========================================================================= */

        /* 1. Hardware Screw Detailing */
        .hw-screw {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4b5563 0%, #1f2937 100%);
            box-shadow: inset 1px 1px 1px rgba(255,255,255,0.35), inset -1px -1px 1px rgba(0,0,0,0.85), 0 1px 2px rgba(0,0,0,0.8);
            position: relative;
            display: inline-block;
        }}
        .hw-screw::after {{
            content: '';
            position: absolute;
            top: 4px;
            left: 2px;
            width: 6px;
            height: 1.5px;
            background: #111827;
            transform: rotate(35deg);
        }}

        /* 2. Machined Metal Panel Plate */
        .panel-plate {{
            background: linear-gradient(180deg, #1c222e 0%, #141822 100%);
            border: 1px solid #2c3647;
            border-radius: 8px;
            box-shadow: 
                0 4px 16px rgba(0, 0, 0, 0.6),
                inset 1px 1px 0 rgba(255, 255, 255, 0.08),
                inset -1px -1px 0 rgba(0, 0, 0, 0.65);
            padding: 1.25rem 1.5rem;
            position: relative;
            margin-bottom: 1.25rem;
        }}

        /* 3. Inset Smoked-Glass Display */
        .glass-display {{
            background: radial-gradient(ellipse at 50% 10%, #0d121c 0%, #05070a 100%);
            border: 1px solid #1c2536;
            border-radius: 6px;
            box-shadow: 
                inset 0 4px 16px rgba(0, 0, 0, 0.95),
                inset 0 0 1px rgba(0, 0, 0, 0.9),
                0 1px 0 rgba(255, 255, 255, 0.05);
            padding: 1rem 1.25rem;
            position: relative;
            overflow: hidden;
        }}
        .glass-display::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 50%, transparent 100%);
            pointer-events: none;
        }}

        /* 4. Physical LED Indicators */
        .hw-led {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            vertical-align: middle;
            border: 1px solid rgba(0, 0, 0, 0.8);
            margin-right: 6px;
            position: relative;
        }}
        .hw-led.green {{
            background: #10b981;
            box-shadow: 0 0 8px #10b981, 0 0 2px #34d399, inset 1px 1px 1px #a7f3d0;
        }}
        .hw-led.amber {{
            background: #f59e0b;
            box-shadow: 0 0 8px #f59e0b, 0 0 2px #fbbf24, inset 1px 1px 1px #fef3c7;
        }}
        .hw-led.red {{
            background: #ef4444;
            box-shadow: 0 0 8px #ef4444, 0 0 2px #f87171, inset 1px 1px 1px #fecaca;
        }}
        .hw-led.blue {{
            background: #38bdf8;
            box-shadow: 0 0 8px #38bdf8, 0 0 2px #7dd3fc, inset 1px 1px 1px #e0f2fe;
        }}
        .hw-led.off {{
            background: #2b3342;
            box-shadow: inset 1px 1px 2px #111827;
        }}

        /* 5. Top Instrument Header Bar */
        .top-instrument-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(180deg, #1c2330 0%, #131822 100%);
            border: 1px solid #2b3548;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.5), inset 1px 1px 0 rgba(255,255,255,0.08), inset -1px -1px 0 rgba(0,0,0,0.6);
            position: relative;
        }}
        .instrument-title {{
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: 0.02em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            text-transform: uppercase;
        }}
        .instrument-subtitle {{
            font-size: 0.8125rem;
            color: var(--text-muted);
            margin: 0.2rem 0 0 0;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* 6. Hardware Status Badges */
        .hw-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: #090c13;
            border: 1px solid #20293a;
            border-radius: 4px;
            padding: 0.3rem 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            color: #cbd5e1;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.7), 0 1px 0 rgba(255,255,255,0.06);
        }}
        .hw-badge.active-amber {{
            border-color: rgba(245, 158, 11, 0.4);
            color: #f59e0b;
        }}
        .hw-badge.active-green {{
            border-color: rgba(16, 185, 129, 0.4);
            color: #10b981;
        }}

        /* 7. Hardware LCD / Segment Metric Gauges */
        .lcd-metric-gauge {{
            background: #06080d;
            border: 1px solid #1a2232;
            border-radius: 6px;
            padding: 1rem 1.25rem;
            box-shadow: inset 0 3px 10px rgba(0,0,0,0.9), 0 1px 0 rgba(255,255,255,0.05);
            position: relative;
            border-top: 2px solid #283347;
        }}
        .lcd-gauge-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6875rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.4rem;
        }}
        .lcd-gauge-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--accent-amber);
            line-height: 1.1;
            letter-spacing: 0.02em;
            text-shadow: 0 0 10px rgba(245, 158, 11, 0.35);
        }}
        .lcd-gauge-subtext {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.35rem;
            font-family: 'Inter', sans-serif;
        }}

        /* 8. Hardware Camera Monitor / Optical Viewport */
        .camera-monitor-bezel {{
            background: #080b11;
            border: 2px solid #1e2638;
            border-radius: 8px;
            padding: 0.75rem;
            box-shadow: 
                inset 0 4px 18px rgba(0, 0, 0, 0.95),
                0 4px 14px rgba(0,0,0,0.6),
                0 1px 0 rgba(255,255,255,0.06);
            position: relative;
            margin-bottom: 1rem;
        }}
        .camera-corner-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6875rem;
            color: #64748b;
            letter-spacing: 0.08em;
            display: flex;
            justify-content: space-between;
            padding: 0.2rem 0.3rem 0.5rem 0.3rem;
            border-bottom: 1px solid #151c2a;
            margin-bottom: 0.5rem;
        }}

        /* 9. Biometric Decision Panels (Match vs Rejected) */
        .decision-panel {{
            background: #080b11;
            border: 1px solid #1f2738;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            position: relative;
            box-shadow: inset 0 3px 12px rgba(0,0,0,0.85);
        }}
        .decision-panel.accepted {{
            border-left: 4px solid #10b981;
            box-shadow: inset 0 3px 12px rgba(0,0,0,0.85), inset 3px 0 12px rgba(16, 185, 129, 0.1);
        }}
        .decision-panel.rejected {{
            border-left: 4px solid #ef4444;
            box-shadow: inset 0 3px 12px rgba(0,0,0,0.85), inset 3px 0 12px rgba(239, 68, 68, 0.1);
        }}

        /* 10. Hardware Similarity Gauge with Threshold Needle */
        .hw-sim-meter {{
            margin: 1rem 0;
            background: #05070a;
            border: 1px solid #18202e;
            border-radius: 4px;
            padding: 0.75rem 1rem;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.8);
        }}
        .hw-meter-labels {{
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #94a3b8;
            margin-bottom: 0.5rem;
        }}
        .hw-meter-track {{
            height: 12px;
            background: #0d121c;
            border: 1px solid #1e2638;
            border-radius: 3px;
            position: relative;
            overflow: visible;
        }}
        .hw-meter-bar {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }}
        .hw-meter-bar.match {{
            background: linear-gradient(90deg, #047857 0%, #10b981 100%);
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
        }}
        .hw-meter-bar.unknown {{
            background: linear-gradient(90deg, #991b1b 0%, #ef4444 100%);
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
        }}
        .hw-needle {{
            position: absolute;
            top: -6px;
            bottom: -6px;
            width: 2px;
            background: #f59e0b;
            box-shadow: 0 0 6px rgba(245, 158, 11, 0.9);
            z-index: 10;
        }}
        .hw-needle-label {{
            position: absolute;
            top: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6875rem;
            font-weight: 700;
            color: #f59e0b;
            transform: translateX(-50%);
            white-space: nowrap;
        }}

        /* 11. Analysis Trace Sequence Step Box */
        .trace-box {{
            background: #080b12;
            border: 1px solid #1a2232;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-top: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #94a3b8;
            line-height: 1.8;
        }}
        .trace-item {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #121824;
            padding: 0.2rem 0;
        }}
        .trace-item:last-child {{
            border-bottom: none;
        }}

        /* 12. Bottom System Console Status Strip */
        .console-status-bar {{
            margin-top: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(180deg, #18202d 0%, #0d1118 100%);
            border: 1px solid #283347;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5), inset 1px 1px 0 rgba(255,255,255,0.06);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.65rem 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #94a3b8;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        /* =========================================================================
           TACTILE STREAMLIT FORM & BUTTON OVERRIDES
           ========================================================================= */

        /* Sidebar Navigation Rocker Switches */
        [data-testid="stSidebar"] div.stButton > button {{
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding: 0.65rem 1rem !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            letter-spacing: 0.01em !important;
            text-transform: none !important;
            border-radius: 6px !important;
            margin-bottom: 0.35rem !important;
            width: 100% !important;
            background: linear-gradient(180deg, #1c2331 0%, #131823 100%) !important;
            border: 1px solid #293448 !important;
            color: #94a3b8 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.07) !important;
        }}
        [data-testid="stSidebar"] div.stButton > button:hover {{
            background: linear-gradient(180deg, #242e40 0%, #18202e 100%) !important;
            border-color: #3d4e6b !important;
            color: #f8fafc !important;
        }}
        /* Active Sidebar Switch State: Tactile Pressed with Amber Glow */
        [data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background: linear-gradient(180deg, #18202d 0%, #0f1420 100%) !important;
            border: 1px solid #f59e0b !important;
            color: #f59e0b !important;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.85), 0 0 12px rgba(245, 158, 11, 0.25) !important;
            font-weight: 700 !important;
            text-shadow: none !important;
        }}
        [data-testid="stSidebar"] div.stButton > button p {{
            text-align: left !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.65rem !important;
            margin: 0 !important;
            width: 100% !important;
            justify-content: flex-start !important;
        }}

        /* Tactile Hardware Buttons */
        div.stButton > button {{
            background: linear-gradient(180deg, #242c3a 0%, #161c26 100%) !important;
            color: #e2e8f0 !important;
            border: 1px solid #364257 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.8125rem !important;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: 0.04em !important;
            padding: 0.55rem 1.25rem !important;
            box-shadow: 
                0 3px 6px rgba(0,0,0,0.4),
                inset 0 1px 0 rgba(255,255,255,0.12),
                inset 0 -1px 0 rgba(0,0,0,0.4) !important;
            transition: all 0.1s ease !important;
        }}
        div.stButton > button:hover {{
            background: linear-gradient(180deg, #2b3546 0%, #1b222f 100%) !important;
            border-color: #4b5b75 !important;
            color: #ffffff !important;
        }}
        div.stButton > button:active {{
            background: linear-gradient(180deg, #121620 0%, #1c2230 100%) !important;
            box-shadow: 
                inset 0 3px 6px rgba(0,0,0,0.7),
                0 1px 0 rgba(255,255,255,0.05) !important;
            transform: translateY(1px) !important;
            border-color: #242c3b !important;
        }}

        /* Primary Tactile Button (Amber Illuminated) */
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(180deg, #d97706 0%, #92400e 100%) !important;
            border: 1px solid #f59e0b !important;
            color: #ffffff !important;
            box-shadow: 
                0 3px 8px rgba(180, 83, 9, 0.5),
                inset 0 1px 0 rgba(254, 243, 199, 0.35),
                inset 0 -1px 0 rgba(0,0,0,0.5) !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background: linear-gradient(180deg, #f59e0b 0%, #b45309 100%) !important;
            border-color: #fbbf24 !important;
        }}
        div.stButton > button[kind="primary"]:active {{
            background: linear-gradient(180deg, #78350f 0%, #92400e 100%) !important;
            box-shadow: inset 0 3px 6px rgba(0,0,0,0.8) !important;
            transform: translateY(1px) !important;
        }}

        /* Inputs & Uploaders */
        div[data-baseweb="input"] > div {{
            background-color: #080b11 !important;
            border: 1px solid #232d3e !important;
            border-radius: 6px !important;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.7) !important;
            color: var(--text-primary) !important;
        }}
        div[data-baseweb="input"] > div:focus-within {{
            border-color: var(--accent-amber) !important;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.7), 0 0 6px rgba(245, 158, 11, 0.3) !important;
        }}
        [data-testid="stFileUploader"] {{
            background-color: #080b11 !important;
            border: 1px dashed #283347 !important;
            border-radius: 8px !important;
            padding: 1.25rem !important;
            text-align: center;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.7) !important;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: var(--accent-amber) !important;
        }}

        /* Hardware Tables & DataFrames */
        [data-testid="stDataFrame"] {{
            border: 1px solid #242c3c !important;
            border-radius: 6px !important;
            overflow: hidden;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.8) !important;
        }}

        /* =========================================================================
           SKEUOMORPHIC WEBCAM & ENROLLMENT WORKBENCH STYLES
           ========================================================================= */

        /* Physical Source Toggle Rocker (Image Upload vs Webcam) */
        .source-rocker-bay {{
            background: #080b11;
            border: 1px solid #1c2536;
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 1.25rem;
            box-shadow: inset 0 3px 8px rgba(0,0,0,0.85);
        }}
        .source-rocker-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6875rem;
            font-weight: 700;
            color: #64748b;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        /* Biometric Webcam Viewfinder Bezel */
        .webcam-viewport-bezel {{
            background: radial-gradient(ellipse at 50% 0%, #0d131f 0%, #05070c 100%);
            border: 2px solid #222d40;
            border-radius: 8px;
            padding: 0.85rem;
            position: relative;
            box-shadow: 
                inset 0 4px 20px rgba(0,0,0,0.95),
                0 6px 20px rgba(0,0,0,0.6),
                inset 0 1px 0 rgba(255,255,255,0.06);
            margin-bottom: 1rem;
        }}
        .webcam-telemetry-header {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #94a3b8;
            letter-spacing: 0.08em;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #151d2c;
            margin-bottom: 0.75rem;
        }}

        /* Biometric Face Alignment Guide Overlay Reticle */
        .camera-reticle-overlay {{
            position: relative;
            width: 100%;
            border-radius: 6px;
            overflow: hidden;
        }}
        .face-target-guide {{
            border: 2px dashed rgba(245, 158, 11, 0.65);
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(245, 158, 11, 0.2), inset 0 0 15px rgba(245, 158, 11, 0.1);
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 220px;
            height: 280px;
            pointer-events: none;
            z-index: 5;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 0.5rem;
        }}
        .face-target-text {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.625rem;
            font-weight: 700;
            color: #f59e0b;
            text-align: center;
            background: rgba(8, 11, 17, 0.75);
            border-radius: 3px;
            padding: 2px 6px;
            letter-spacing: 0.08em;
            align-self: center;
        }}

        /* Streamlit Camera Input Hardware Bezel Integration */
        [data-testid="stCameraInput"] {{
            background-color: #06080e !important;
            border: 1px solid #1b2332 !important;
            border-radius: 6px !important;
            padding: 0.75rem !important;
            box-shadow: inset 0 3px 10px rgba(0,0,0,0.85) !important;
        }}
        [data-testid="stCameraInput"] video {{
            border-radius: 6px !important;
            border: 1px solid #2a364a !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.6) !important;
        }}
        [data-testid="stCameraInput"] button {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
        }}

        /* Guidance Checklist Pill */
        .camera-guidance-pill {{
            background: #090c13;
            border: 1px solid #1c2434;
            border-radius: 4px;
            padding: 0.4rem 0.65rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .camera-guidance-pill.valid {{
            color: #10b981;
            border-color: rgba(16, 185, 129, 0.3);
        }}
        .camera-guidance-pill.warning {{
            color: #f59e0b;
            border-color: rgba(245, 158, 11, 0.3);
        }}

        /* Responsive overrides */
        @media (max-width: 1024px) {{
            .block-container {{
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }}
            .console-status-bar {{
                font-size: 0.6rem;
                padding: 0 0.75rem;
            }}
        }}
    </style>
    """


def clean_html(html_str: str) -> str:
    """Strip leading and trailing whitespace from every line of HTML so markdown never parses it as a code block."""
    if not html_str:
        return ""
    lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    return "".join(lines)


def render_html(html_str: str):
    """Render HTML safely in Streamlit without code-block leakage."""
    import streamlit as st
    st.markdown(clean_html(html_str), unsafe_allow_html=True)
