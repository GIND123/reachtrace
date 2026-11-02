"""Exact metagradients through the weighted CPT run (synthetic/magic.py), on a tiny model."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from reachtrace.synthetic import magic  # noqa: E402
from reachtrace.synthetic.cpt import CPTConfig, train_cpt  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    cfg = transformers.LlamaConfig(
        vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(cfg).save_pretrained(path)
    rng = np.random.default_rng(0)
    docs = [rng.integers(1, 48, rng.integers(5, 12)).tolist() for _ in range(10)]
    rows = []
    for _ in range(3):
        ids = rng.integers(1, 48, 9).tolist()
        rows.append({"input_ids": ids, "labels": [-100] * 4 + ids[4:]})
    cfg = CPTConfig(lr=3e-2, warmup_steps=2, epochs=2, batch_docs=4, max_len=16,
                    weight_decay=0.1, micro_docs=2)
    return str(path), docs, rows, cfg


def test_replay_matches_train_cpt(tiny):
    path, docs, _, cfg = tiny
    a, _ = train_cpt(path, docs, np.ones(len(docs)), cfg, device="cpu")
    b = magic.replay(path, docs, np.ones(len(docs)), cfg, device="cpu")
    for (k, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(x, y), k


def test_metagradient_matches_finite_differences(tiny, tmp_path):
    # GPT-2 keeps every operation in float64 (Llama's RMSNorm computes in float32, which
    # limits finite differences to ~1e-3), so the check can be tight.
    _, docs, rows, cfg = tiny
    path = str(tmp_path / "gpt2")
    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=48, n_embd=16, n_layer=2, n_head=2, n_positions=32, resid_pdrop=0,
        embd_pdrop=0, attn_pdrop=0)).save_pretrained(path)
    coef = np.array([[1.0, 0.0], [-0.5, 1.0], [0.25, 1.0]])
    f64 = torch.float64
    # With Adam's eps at 1e-8 the update m / (sqrt(v) + eps) bends sharply where gradient
    # entries are near eps, so finite differences converge slowly; eps 1e-3 is smooth.
    for eps, rtol in [(1e-3, 1e-6), (1e-8, 1e-4)]:
        c = dataclasses.replace(cfg, adam_eps=eps)
        scores, info = magic.metagradients(path, docs, c, rows, coef, device="cpu",
                                           dtype=f64, segment=2, hvp_docs=3,
                                           log=lambda *_: None)

        def J(w, c=c):
            model = magic.replay(path, docs, w, c, device="cpu", dtype=f64)
            return magic._target_grads(model, rows, coef, "cpu")[1]

        assert np.allclose(info["J"], J(np.ones(len(docs))))
        h = 1e-6
        for d in [0, 3, 7]:
            up, dn = np.ones(len(docs)), np.ones(len(docs))
            up[d] += h
            dn[d] -= h
            fd = (J(up) - J(dn)) / (2 * h)
            assert np.allclose(scores[d], fd, rtol=rtol, atol=1e-9), (eps, d, scores[d], fd)
