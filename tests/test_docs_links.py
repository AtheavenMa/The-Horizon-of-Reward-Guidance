"""Check the checked-in bilingual documentation and published media are linked locally."""
import re
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
PAGES=[ROOT/'README.md',ROOT/'README_cn.md',*sorted((ROOT/'docs').glob('*.md')),
       ROOT/'THIRD_PARTY_NOTICES.md',ROOT/'CONTRIBUTING.md',ROOT/'CHANGELOG.md']


def find_local_links(markdown:str):
    # Markdown targets and explicit HTML image/link attributes.
    md=re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',markdown)
    md+=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',markdown)
    md+=re.findall(r'(?:src|href)="([^"]+)"',markdown)
    for link in md:
        link=link.strip()
        if link.startswith(('https://','http://','mailto:','#','data:')):
            continue
        yield link.split('#',1)[0].split('?',1)[0]


@pytest.mark.parametrize('page',PAGES,ids=lambda p:str(p.relative_to(ROOT)))
def test_local_links(page:Path):
    text=page.read_text(encoding='utf8')
    for link in find_local_links(text):
        assert (page.parent/link).exists(),f'{page.relative_to(ROOT)}: dead link {link}'


def test_preprint_pending_and_all_figure_assets_exist():
    from experiments.reproduce import REFERENCE
    assert not (ROOT/'paper').exists(), 'Unreleased manuscript must not be bundled'
    assert 'arXiv coming soon' in (ROOT/'README.md').read_text(encoding='utf8')
    assert 'arXiv coming soon' in (ROOT/'README_cn.md').read_text(encoding='utf8')
    legacy_figures = {
        "actual_adaptation.png", "closed_loop.png", "discrete_resources.png",
        "full_support_separation.png", "instance_complexity.png",
        "ordinary_pg.png", "query_energy.png",
    }
    assert legacy_figures <= {p.name for p in (ROOT/'assets/figures').glob('*.png')}
    assert len(list(REFERENCE.glob('*.json')))==4
    assert len(list(REFERENCE.glob('*.csv')))==5


def test_main_result_figures_are_available_as_vector_raster_and_pdf():
    root = ROOT / "assets" / "figures"
    for number in range(2, 7):
        for suffix in (".png", ".svg", ".pdf"):
            path = root / f"fig{number}{suffix}"
            assert path.exists() and path.stat().st_size > 1_000, path
        svg = (root / f"fig{number}.svg").read_text(encoding="utf-8")
        assert "<text" in svg and "<svg" in svg

    for homepage in (ROOT / "README.md", ROOT / "README_cn.md"):
        text = homepage.read_text(encoding="utf-8")
        assert text.count("<details>") == 5
        for number in range(2, 7):
            assert f"assets/figures/fig{number}.png" in text
        assert "source-only" not in text.lower()
