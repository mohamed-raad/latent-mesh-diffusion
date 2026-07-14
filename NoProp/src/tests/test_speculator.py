import json
import os
import tempfile
import pytest
import torch
import sys

sys.path.insert(0, "NoProp/src")

from dspark_speculator import (
    MTPHead,
    MultiTokenPredictor,
    ConfidenceVerifier,
    DSparkSpeculator,
    CurriculumDataset,
)


def test_mtp_head():
    head = MTPHead(embed_dim=64, vocab_size=1000)
    x = torch.randn(2, 64)
    logits = head(x)
    assert logits.shape == (2, 1000)


def test_multi_token_predictor():
    predictor = MultiTokenPredictor(embed_dim=64, vocab_size=1000, num_draft_tokens=3)
    hidden = torch.randn(2, 64)
    logits_seq = predictor(hidden)
    assert len(logits_seq) == 3
    assert all(l.shape == (2, 1000) for l in logits_seq)


def test_multi_token_draft():
    predictor = MultiTokenPredictor(embed_dim=64, vocab_size=1000, num_draft_tokens=3)
    hidden = torch.randn(2, 64)
    tokens = predictor.draft(hidden)
    assert tokens.shape == (2, 3)
    assert tokens.dtype == torch.long


def test_confidence_verifier():
    verifier = ConfidenceVerifier(embed_dim=64)
    hidden = torch.randn(2, 64)
    conf = verifier(hidden)
    assert conf.shape == (2, 1)
    assert (conf >= 0).all() and (conf <= 1).all()


def test_dspark_speculate():
    spec = DSparkSpeculator(embed_dim=64, vocab_size=1000, num_draft_tokens=3, confidence_threshold=0.5)
    hidden = torch.randn(2, 64)
    accepted, conf = spec.speculate(hidden)
    assert accepted.shape == (2, 3)
    assert conf.shape == (2, 1)


def test_dspark_loss():
    spec = DSparkSpeculator(embed_dim=64, vocab_size=1000, num_draft_tokens=3)
    hidden = torch.randn(2, 64)
    targets = torch.randint(0, 1000, (2, 3))
    loss_val = spec.loss(hidden, targets)
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.item() >= 0.0


def test_curriculum_dataset_empty_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="No curriculum samples found"):
            CurriculumDataset(data_dir=tmp)


def test_curriculum_dataset_single_sample():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "phase_0.jsonl")
        with open(fpath, "w") as f:
            f.write(json.dumps({"tokens": [1, 2, 3], "meta": {"domain": "math"}}) + "\n")
        ds = CurriculumDataset(data_dir=tmp, max_seq_len=512)
        assert len(ds) == 1
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert "meta" in sample
        assert sample["input_ids"].tolist() == [1, 2, 3]


def test_curriculum_dataset_truncation():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "phase_0.jsonl")
        with open(fpath, "w") as f:
            f.write(json.dumps({"tokens": list(range(100))}) + "\n")
        ds = CurriculumDataset(data_dir=tmp, max_seq_len=10)
        sample = ds[0]
        assert sample["input_ids"].shape[0] == 10


def test_curriculum_dataset_phase_filter():
    with tempfile.TemporaryDirectory() as tmp:
        for phase in [0, 1, 2]:
            fpath = os.path.join(tmp, f"phase_{phase}.jsonl")
            with open(fpath, "w") as f:
                f.write(json.dumps({"tokens": [phase]}) + "\n")
        ds = CurriculumDataset(data_dir=tmp, phases=[0, 2])
        assert len(ds) == 2


def test_curriculum_dataset_skip_empty_lines():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "phase_0.jsonl")
        with open(fpath, "w") as f:
            f.write("\n")
            f.write(json.dumps({"tokens": [1, 2, 3]}) + "\n")
            f.write("\n")
        ds = CurriculumDataset(data_dir=tmp)
        assert len(ds) == 1


def test_curriculum_dataset_skip_invalid_json():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "phase_0.jsonl")
        with open(fpath, "w") as f:
            f.write("{invalid json}\n")
            f.write(json.dumps({"tokens": [1, 2, 3]}) + "\n")
        ds = CurriculumDataset(data_dir=tmp)
        assert len(ds) == 1


def test_mtp_loss_with_curriculum_weight():
    predictor = MultiTokenPredictor(embed_dim=64, vocab_size=1000, num_draft_tokens=3)
    hidden = torch.randn(2, 64)
    targets = torch.randint(0, 1000, (2, 3))
    loss_unweighted = predictor.loss(hidden, targets, curriculum_weight=1.0)
    loss_weighted = predictor.loss(hidden, targets, curriculum_weight=2.0)
    assert loss_weighted.item() > loss_unweighted.item()
    ratio = loss_weighted.item() / max(loss_unweighted.item(), 1e-8)
    assert 1.5 < ratio < 2.5
