from pathlib import Path

from tsfoundation.data.dataset import load_dpo_dataset, load_sft_dataset
from tsfoundation.data.serialization import parse_series


ROOT = Path(__file__).resolve().parents[1]


def test_sft_sample_loads():
    records = load_sft_dataset([ROOT / "data/sample_sft/energy_sample.json"])
    assert len(records) == 2
    assert len(parse_series(records[0]["input"])) == 12
    assert len(parse_series(records[0]["output"])) == 3


def test_dpo_sample_loads():
    records = load_dpo_dataset([ROOT / "data/sample_dpo/target_domain_dpo_sample.json"])
    assert len(records) == 2
    assert len(parse_series(records[0]["chosen"])) == 3
    assert len(parse_series(records[0]["rejected"])) == 3

