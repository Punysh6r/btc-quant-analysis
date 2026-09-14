"""Design assets: dark terminal theme parses and the CSS hooks exist."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_theme_is_dark_terminal():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    theme = cfg["theme"]
    assert theme["base"] == "dark"
    assert theme["backgroundColor"] == "#0b0f14"
    assert theme["primaryColor"] == "#22c55e"
    assert theme["textColor"] == "#e6edf3"


def test_terminal_css_covers_key_hooks():
    css = (ROOT / "assets" / "terminal.css").read_text(encoding="utf-8")
    for hook in ('[data-testid="stMetric"]', '[data-testid="stSidebar"]', "h1"):
        assert hook in css


def test_app_loads_terminal_css():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "_load_terminal_css()" in src
    assert "assets" in src and "terminal.css" in src
