import json
import subprocess
import sys
from pathlib import Path
import pytest
from experiments.reproduce import assert_equivalent

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_numerical_ledger():
    reference=json.loads((ROOT/"results/reference/paper_results_v10.json").read_text())
    assert reference["inherited_v06"]["five_output_closed_loop"]["source_only_exact_deadline"]==pytest.approx(5.)
    assert reference["inherited_v06"]["five_output_closed_loop"]["post_query_min_certificate_normalized"]==pytest.approx(.682)
    assert len(reference["added_discrete_resource_experiment"]["runs"])==4
    assert_equivalent(reference,reference)


@pytest.mark.parametrize("script,summary,field",[
    ("demo.py","outputs/demo_horizon/summary.json","support_horizon"),
    ("demo_feedback.py","outputs/demo_feedback/summary.json","post_query_min_certificate"),
    ("demo_ordinary_pg.py","outputs/demo_ordinary_pg/summary.json","observed_maximal_speed_contact"),
])
def test_demos_execute(tmp_path:Path,script:str,summary:str,field:str):
    out=tmp_path/script.replace(".py","")
    subprocess.run([sys.executable,str(ROOT/"demo"/script),"--out",str(out)],
                   check=True,cwd=ROOT,capture_output=True,text=True)
    data=json.loads((out/"summary.json").read_text())
    assert field in data
    assert (out/"demo_output.png").stat().st_size>1000


def test_preprint_manuscript_is_not_distributed():
    assert not (ROOT/"paper").exists()
    assert "arXiv coming soon" in (ROOT/"README.md").read_text(encoding="utf-8")
