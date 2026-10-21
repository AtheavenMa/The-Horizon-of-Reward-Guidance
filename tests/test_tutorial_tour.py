"""Smoke checks for the seven-slide, static five-minute research tutorial."""
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]


def test_tour_is_seven_slides_with_notes_and_theory_details():
    html = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    assert html.count('<section class="slide ') == 7
    assert html.count('class="notes"') == 7
    assert html.count('class="drawer"') == 6
    assert 'class SlidePresentation' in html
    assert '<script src=' not in html
    assert "RSS in Five Minutes" not in html
    assert "The Horizon of Reward Guidance" not in html
    assert "The Limits of <em>Reward Guidance</em>" in html
    for control in ("deltaSlider", "timeSlider", "rhoChoices", "feedbackChoices", "studyDst", "studyPythia"):
        assert f'id="{control}"' in html
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids))


def test_tutorial_is_presented_like_sar_and_rss():
    preview = ROOT / "assets/tour_preview.png"
    assert preview.is_file()
    raw = preview.read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", raw[16:24]) == (1920, 360)
    assert (ROOT / "docs/.nojekyll").exists()
    for homepage in ("README.md", "README_cn.md"):
        text = (ROOT / homepage).read_text(encoding="utf-8")
        assert "5-" in text or "5 分钟" in text
        assert "assets/tour_preview.png" in text
        assert "docs/index.html" in text
        assert "The-Horizon-of-Reward-Guidance" not in text
