"""View implementations for VISION ID dashboard."""

from app.ui.views.home_view import render_home_view
from app.ui.views.dashboard_view import render_dashboard_view
from app.ui.views.identify_view import render_identify_view
from app.ui.views.enroll_view import render_enroll_view
from app.ui.views.people_view import render_people_view
from app.ui.views.evaluation_view import render_evaluation_view
from app.ui.views.settings_view import render_settings_view

__all__ = [
    "render_home_view",
    "render_dashboard_view",
    "render_identify_view",
    "render_enroll_view",
    "render_people_view",
    "render_evaluation_view",
    "render_settings_view",
]
