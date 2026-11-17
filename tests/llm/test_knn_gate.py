"""Tests for kNN-LM gate fitting (docs/KNN_LM.md, PR 3).

Gate criteria:
- The cached fused target log-prob equals what inference computes
  (``fused_logprobs``) for the same store, position and gate, so the gate
  is fitted on exactly the quantity it's later used for.
- ``collect_features`` scores answer tokens only, conditioned on the full
  prompt, skips examples without a context block, and truncates the answer
  (never the prompt) to fit ``max_seq_len``.
- ``split_by_example`` never puts one example's positions on both sides.
- Fitting learns to lean on the store when it predicts the target, and to
  back off when it never does.
- ``gate.features`` ignores ``+inf`` padding.
- The CLI script runs end to end, excludes the quiz and ``--exclude``
  questions, and writes a loadable gate.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from grimoire_ai.llm.inference.knn_lm import (
    KNNConfig,
    LambdaGate,
    ScopedStore,
    context_span,
    fused_logprobs,
    load_gate,
)
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import AST_ID, BOS_ID, EOS_ID, SEP_ID, USR_ID
from grimoire_ai.llm.training.knn_gate import (
    GateExample,
    GateFeatures,
    collect_features,
    fit_gate,
    fused_target_logp,
    gate_report,
    split_by_example,
)

CFG = KNNConfig(k=4)


def _tiny_model(max_seq_len: int = 48) -> GrimoireTransformer:
    torch.manual_seed(0)
    cfg = TransformerConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        d_ff=64, max_seq_len=max_seq_len, dropout=0.0,
    )
    return GrimoireTransformer(cfg).eval()


def _example(ctx: list[int], answer: list[int]) -> GateExample:
    return GateExample([BOS_ID, SEP_ID, *ctx, SEP_ID, USR_ID, 40, 41, AST_ID], answer)


def _trained_looking_gate() -> LambdaGate:
    gate = LambdaGate(lam_max=0.8, init_temperature=3.0)
    with torch.no_grad():
        gate.lin.weight.copy_(torch.tensor([[-0.3, 0.1, 0.2]]))
        gate.lin.bias.fill_(0.4)
    return gate


# ------------------------------------------------------- collect_features


def test_cached_logp_matches_inference_fusion() -> None:
    model = _tiny_model()
    ex = _example(list(range(10, 22)), [12, 13, 14, 30, EOS_ID])
    f = collect_features(model, [ex], CFG)
    gate = _trained_looking_gate()
    cached, _ = fused_target_logp(gate, f)

    # Recompute the inference way: same store, per-position fused_logprobs.
    ids = ex.prompt_ids + ex.answer_ids
    captured = {}
    h = model.final_norm.register_forward_hook(lambda _m, _i, o: captured.setdefault("h", o))
    with torch.no_grad():
        logits = model(torch.tensor([ids]))
    h.remove()
    hidden = captured["h"][0]
    store = ScopedStore(hidden.size(-1))
    store.add_transitions(hidden, ids, *context_span(ex.prompt_ids))
    P = len(ex.prompt_ids)
    with torch.no_grad():
        expected = torch.stack([
            fused_logprobs(logits[0, t][None], hidden[t][None], store, gate, CFG)[0, ids[t + 1]]
            for t in range(P - 1, len(ids) - 1)
        ])
    assert torch.allclose(cached.detach(), expected, atol=1e-4)


def test_collect_scores_answer_tokens_with_prompt_context() -> None:
    model = _tiny_model()
    ex = _example(list(range(10, 22)), [12, 13, EOS_ID])
    # k above the store size (12 keys): every key is a neighbor, so "hit"
    # means exactly "target appears in the context".
    f = collect_features(model, [ex], KNNConfig(k=16))

    assert f.n_positions == 3 and f.n_examples == 1
    ids = ex.prompt_ids + ex.answer_ids
    with torch.no_grad():
        lp = model(torch.tensor([ids]))[0].log_softmax(-1)
    P = len(ex.prompt_ids)
    manual = torch.stack([lp[P - 1 + i, tok] for i, tok in enumerate(ex.answer_ids)])
    assert torch.allclose(f.lm_target_logp, manual, atol=1e-5)
    # Targets 12 and 13 are context tokens, so some neighbor must hold them;
    # EOS never appears in the context.
    assert f.hit[0].any() and f.hit[1].any() and not f.hit[2].any()


def test_collect_skips_no_context_and_truncates_answer_only() -> None:
    model = _tiny_model(max_seq_len=24)
    no_ctx = GateExample([BOS_ID, USR_ID, 40, AST_ID], [12, EOS_ID])
    long_answer = _example(list(range(10, 22)), list(range(10, 40)))  # prompt is 18 tokens
    f = collect_features(model, [no_ctx, long_answer], CFG)

    assert f.n_skipped == 1 and f.n_examples == 1
    assert f.n_positions == 24 - len(long_answer.prompt_ids)


def test_collect_pads_small_stores_and_features_ignore_padding() -> None:
    model = _tiny_model()
    f = collect_features(model, [_example([10, 11], [10, EOS_ID])], KNNConfig(k=8))
    assert f.D.shape == (2, 8)
    assert torch.isinf(f.D[:, 2:]).all() and not f.hit[:, 2:].any()
    assert torch.isfinite(f.feats).all()

    D = torch.tensor([[1.0, 3.0, float("inf")]])
    feats = LambdaGate.features(D, torch.zeros(1, 4).log_softmax(-1))
    assert feats[0, 1].item() == pytest.approx(torch.log1p(torch.tensor(2.0)).item())


def test_collect_raises_when_nothing_usable() -> None:
    with pytest.raises(ValueError, match="no usable examples"):
        collect_features(_tiny_model(), [GateExample([BOS_ID, USR_ID, AST_ID], [EOS_ID])], CFG)


# ------------------------------------------------------- split and fitting


def _synthetic(n_examples: int, per_example: int, knn_predicts: bool) -> GateFeatures:
    """LM gives the target logp -4; the store's nearest neighbor holds the
    target (or never does), at a small distance."""
    torch.manual_seed(0)
    N, k = n_examples * per_example, 4
    D = torch.sort(torch.rand(N, k) * 10 + 1, dim=-1).values
    hit = torch.zeros(N, k, dtype=torch.bool)
    if knn_predicts:
        hit[:, 0] = True
    lm_lp = torch.full((N, 8), -4.0)
    return GateFeatures(
        D=D, hit=hit, lm_target_logp=torch.full((N,), -4.0),
        feats=LambdaGate.features(D, lm_lp.log_softmax(-1)),
        example=torch.arange(n_examples).repeat_interleave(per_example),
        n_examples=n_examples, n_skipped=0,
    )


def test_split_by_example_is_disjoint() -> None:
    f = _synthetic(10, 5, True)
    train, val = split_by_example(f, val_frac=0.3, seed=1)
    assert (train ^ val).all()
    assert set(f.example[train].tolist()).isdisjoint(f.example[val].tolist())
    assert len(set(f.example[val].tolist())) == 3
    with pytest.raises(ValueError):
        split_by_example(f, val_frac=1.0)


def test_fit_leans_on_store_when_it_predicts_the_target() -> None:
    f = _synthetic(10, 20, knn_predicts=True)
    train, val = split_by_example(f)
    gate = fit_gate(f, train, CFG, epochs=200)
    rep = gate_report(gate, f, val)
    assert rep["fused_nll"] < rep["lm_nll"] - 1.0
    assert rep["mean_lambda"] > 0.5 and rep["coverage"] == 1.0


def test_fit_backs_off_when_store_never_helps() -> None:
    f = _synthetic(10, 20, knn_predicts=False)
    train, val = split_by_example(f)
    gate = fit_gate(f, train, CFG, epochs=300)
    rep = gate_report(gate, f, val)
    assert rep["mean_lambda"] < 0.05
    assert rep["delta_nll"] < 0.05  # costs (almost) nothing once backed off
    assert rep["coverage"] == 0.0


# ------------------------------------------------------------------ script


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fit_knn_gate",
        Path(__file__).resolve().parents[2] / "scripts" / "eval" / "fit_knn_gate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_script_end_to_end_with_record_context(tmp_path, monkeypatch, capsys) -> None:
    from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
    from grimoire_ai.llm.training.checkpoint import save_checkpoint

    cfg = TransformerConfig(
        vocab_size=300, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        d_ff=64, max_seq_len=128, dropout=0.0,
    )
    model = GrimoireTransformer(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(path=str(ckpt), model=model,
                    optimizer=torch.optim.AdamW(model.parameters()),
                    step=1, config_dict=cfg.to_dict(), train_loss=4.0)
    tok = tmp_path / "bpe.json"
    enc = BytePairEncoder()
    enc.train(["a grappled creature has its speed reduced to zero " * 30], vocab_size=300)
    enc.save(str(tok))

    rows = [{"user": f"question {i} about grapple", "assistant": "the creature has speed zero",
             "context": "a grappled creature has its speed reduced to zero"} for i in range(8)]
    rows.append({"user": "quiz question", "assistant": "x", "context": "y"})
    rows.append({"user": "sft question", "assistant": "x", "context": "y"})
    rows.append({"user": "copied", "assistant": "speed reduced to zero",
                 "context": "its Speed reduced  to zero today"})
    data, quiz, sft = tmp_path / "data.jsonl", tmp_path / "quiz.jsonl", tmp_path / "sft.jsonl"
    _write_jsonl(data, rows)
    _write_jsonl(quiz, [{"user": "Quiz  Question", "assistant": "x"}])
    _write_jsonl(sft, [{"user": "sft question", "assistant": "x"}])
    out = tmp_path / "gate.pt"

    monkeypatch.setattr(sys, "argv", [
        "fit_knn_gate.py", "--checkpoint", str(ckpt), "--vocab", str(tok),
        "--data", str(data), "--quiz", str(quiz), "--exclude", str(sft),
        "--k", "4", "--epochs", "20", "--val-frac", "0.25", "--out", str(out),
    ])
    with pytest.raises(SystemExit) as exc:
        _load_script().main()
    assert exc.value.code in (0, 2)  # verdict depends on random weights

    printed = capsys.readouterr().out
    assert "11 loaded, 2 dropped" in printed  # quiz (case/space-insensitive) + SFT
    assert "Dropped 1 record(s) whose answer appears verbatim" in printed
    assert "VERDICT:" in printed
    gate, loaded_cfg, _ = load_gate(out)
    assert loaded_cfg.k == 4
