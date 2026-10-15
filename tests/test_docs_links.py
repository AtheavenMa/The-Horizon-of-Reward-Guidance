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
    assert len(list((ROOT/'assets/figures').glob('*.png')))==7
    assert len(list(REFERENCE.glob('*.json')))==4
    assert len(list(REFERENCE.glob('*.csv')))==5
