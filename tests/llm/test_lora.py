"""Unit tests for LoRA / adapter fine-tuning.

Coverage:
    LoRALinear          — forward output, zero-init delta, merge()
    add_lora_adapters   — parameter freezing, correct modules replaced,
                          trainable count, num_parameters()
    merge_and_unload    — equivalence of merged vs un-merged output,
                          all params unfrozen after merge
    save_lora / load_lora — round-trip serialisation
    InferenceEngine.load_lora — apply adapter post-load
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.lora import LoRALinear, load_lora, save_lora
from grimoire_ai.llm.model.transformer import GrimoireTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


def _small_model() -> GrimoireTransformer:
    return GrimoireTransformer(_small_config())


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_initial_delta_is_zero(self):
        """With B=0, LoRALinear output == base Linear output."""
        linear = nn.Linear(16, 8, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(2, 16)
        with torch.no_grad():
            expected = linear(x)
            got = lora(x)
        assert torch.allclose(got, expected, atol=1e-5)

    def test_forward_adds_lora_delta(self):
        """After setting non-zero B, output differs from base."""
        linear = nn.Linear(16, 8, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        nn.init.normal_(lora.lora_B)  # break the zero init
        x = torch.randn(2, 16)
        with torch.no_grad():
            base_out = nn.functional.linear(x, lora.base_weight)
            got = lora(x)
        assert not torch.allclose(got, base_out, atol=1e-5)

    def test_scale_is_alpha_over_rank(self):
        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=12.0)
        assert lora.scale == pytest.approx(3.0)

    def test_base_weight_not_trainable(self):
        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        assert "base_weight" not in {n for n, _ in lora.named_parameters()}

    def test_lora_params_are_trainable(self):
        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        param_names = {n for n, _ in lora.named_parameters()}
        assert "lora_A" in param_names
        assert "lora_B" in param_names

    def test_lora_B_initialised_to_zero(self):
        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        assert lora.lora_B.data.abs().max().item() == 0.0

    def test_merge_produces_correct_weight(self):
        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=2, alpha=4.0)
        nn.init.normal_(lora.lora_B)
        merged = lora.merge()
        expected_weight = lora.base_weight + (lora.lora_B @ lora.lora_A) * lora.scale
        assert torch.allclose(merged.weight.data, expected_weight, atol=1e-5)

    def test_merge_output_matches_lora_output(self):
        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=2, alpha=4.0)
        nn.init.normal_(lora.lora_B)
        merged = lora.merge()
        x = torch.randn(3, 8)
        with torch.no_grad():
            assert torch.allclose(lora(x), merged(x), atol=1e-5)

    def test_with_bias(self):
        linear = nn.Linear(8, 4, bias=True)
        lora = LoRALinear(linear, rank=2, alpha=4.0)
        x = torch.randn(2, 8)
        with torch.no_grad():
            out = lora(x)
        assert out.shape == (2, 4)
        merged = lora.merge()
        assert merged.bias is not None

    def test_device_move(self):
        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=2, alpha=4.0)
        # Should not crash; buffers and params move together.
        lora_cpu = lora.to("cpu")
        assert lora_cpu.base_weight.device.type == "cpu"


# ---------------------------------------------------------------------------
# GrimoireTransformer.add_lora_adapters
# ---------------------------------------------------------------------------

class TestAddLoraAdapters:
    def test_target_modules_replaced(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        for block in model.blocks:
            assert isinstance(block.attn.q_proj, LoRALinear)
            assert isinstance(block.attn.v_proj, LoRALinear)
            assert not isinstance(block.attn.k_proj, LoRALinear)
            assert not isinstance(block.attn.o_proj, LoRALinear)

    def test_non_target_modules_unchanged(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj"])
        for block in model.blocks:
            assert isinstance(block.attn.k_proj, nn.Linear)
            assert isinstance(block.attn.v_proj, nn.Linear)
            assert isinstance(block.ffn.gate_proj, nn.Linear)

    def test_base_weights_frozen(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0)
        frozen = [
            p for n, p in model.named_parameters()
            if "lora_A" not in n and "lora_B" not in n
        ]
        assert all(not p.requires_grad for p in frozen)

    def test_lora_params_trainable(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0)
        lora_params = [
            p for n, p in model.named_parameters()
            if "lora_A" in n or "lora_B" in n
        ]
        assert len(lora_params) > 0
        assert all(p.requires_grad for p in lora_params)

    def test_trainable_count_is_small(self):
        model = _small_model()
        total = model.num_parameters(trainable_only=False)
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        trainable = model.num_parameters(trainable_only=True)
        # With rank=4 on q_proj + v_proj across 2 layers, trainable << total.
        assert trainable < total // 10

    def test_ffn_targets(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["gate_proj", "down_proj"])
        for block in model.blocks:
            assert isinstance(block.ffn.gate_proj, LoRALinear)
            assert isinstance(block.ffn.down_proj, LoRALinear)
            assert not isinstance(block.ffn.up_proj, LoRALinear)

    def test_default_targets_are_q_and_v(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0)
        for block in model.blocks:
            assert isinstance(block.attn.q_proj, LoRALinear)
            assert isinstance(block.attn.v_proj, LoRALinear)
            assert isinstance(block.attn.k_proj, nn.Linear)
            assert isinstance(block.attn.o_proj, nn.Linear)


# ---------------------------------------------------------------------------
# GrimoireTransformer.merge_and_unload
# ---------------------------------------------------------------------------

class TestMergeAndUnload:
    def test_output_unchanged_after_merge(self):
        model = _small_model()
        model.eval()
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        # Give lora_B a non-zero value so the delta is real.
        for n, p in model.named_parameters():
            if "lora_B" in n:
                nn.init.normal_(p)
        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out_lora = model(ids)
        model.merge_and_unload()
        with torch.no_grad():
            out_merged = model(ids)
        assert torch.allclose(out_lora, out_merged, atol=1e-4)

    def test_no_lora_modules_after_merge(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0)
        model.merge_and_unload()
        assert not any(isinstance(m, LoRALinear) for m in model.modules())

    def test_all_params_trainable_after_merge(self):
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0)
        model.merge_and_unload()
        assert all(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# save_lora / load_lora round-trip
# ---------------------------------------------------------------------------

class TestLoraSerialisation:
    def test_save_and_load_roundtrip(self):
        import copy
        # Create base model and remember its weights before LoRA.
        model = _small_model()
        base_sd = copy.deepcopy(model.state_dict())

        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        for n, p in model.named_parameters():
            if "lora_B" in n:
                nn.init.normal_(p)

        ids = torch.randint(0, 256, (1, 8))
        model.eval()
        with torch.no_grad():
            out_before = model(ids)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.lora")
            save_lora(model, rank=4, alpha=8.0, targets=["q_proj", "v_proj"], path=path)

            # Fresh model with the SAME base weights.
            model2 = _small_model()
            model2.load_state_dict(base_sd)
            payload = load_lora(model2, path)

            assert payload["rank"] == 4
            assert payload["alpha"] == 8.0
            assert set(payload["targets"]) == {"q_proj", "v_proj"}

            model2.eval()
            with torch.no_grad():
                out_after = model2(ids)

        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_load_into_model_with_existing_adapters(self):
        """load_lora on a model that already has adapters should still work."""
        import copy
        model = _small_model()
        base_sd = copy.deepcopy(model.state_dict())

        model.add_lora_adapters(rank=4, alpha=8.0)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                nn.init.normal_(p)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.lora")
            save_lora(model, rank=4, alpha=8.0, targets=["q_proj", "v_proj"], path=path)

            # model2 has same base weights but fresh (zero) LoRA.
            model2 = _small_model()
            model2.load_state_dict(base_sd)
            model2.add_lora_adapters(rank=4, alpha=8.0)
            load_lora(model2, path)

        ids = torch.randint(0, 256, (1, 8))
        model.eval(); model2.eval()
        with torch.no_grad():
            assert torch.allclose(model(ids), model2(ids), atol=1e-5)

    def test_lora_file_is_small(self):
        """A .lora file should be much smaller than a full checkpoint."""
        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0)
        with tempfile.TemporaryDirectory() as tmp:
            lora_path = Path(tmp) / "test.lora"
            full_path = Path(tmp) / "test.pt"
            save_lora(model, rank=4, alpha=8.0, targets=["q_proj", "v_proj"], path=str(lora_path))
            torch.save({"model": model.state_dict()}, str(full_path))
            assert lora_path.stat().st_size < full_path.stat().st_size


# ---------------------------------------------------------------------------
# InferenceEngine.load_lora
# ---------------------------------------------------------------------------

class TestEngineLoadLora:
    def test_load_lora_applies_adapter(self):
        """Engine.load_lora() should produce same output as the LoRA training model."""
        import copy
        from grimoire_ai.llm.inference.engine import InferenceEngine
        from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
        from grimoire_ai.llm.training.checkpoint import save_checkpoint

        config = _small_config()

        # base model (no LoRA) — this is what gets saved as the checkpoint.
        base_model = _small_model()
        base_sd = copy.deepcopy(base_model.state_dict())

        # LoRA model with the SAME base weights + a trained delta.
        lora_model = _small_model()
        lora_model.load_state_dict(base_sd)
        lora_model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        for n, p in lora_model.named_parameters():
            if "lora_B" in n:
                nn.init.normal_(p)
        lora_model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path  = str(Path(tmp) / "base.pt")
            lora_path  = str(Path(tmp) / "test.lora")
            vocab_path = str(Path(tmp) / "bpe.json")

            save_checkpoint(
                path=ckpt_path,
                model=base_model,
                optimizer=torch.optim.SGD(base_model.parameters(), lr=1e-3),
                step=0,
                config_dict=config.to_dict(),
            )
            save_lora(lora_model, rank=4, alpha=8.0,
                      targets=["q_proj", "v_proj"], path=lora_path)

            enc = BytePairEncoder()
            enc.train(["hello world " * 50], vocab_size=262)
            enc.save(vocab_path)

            engine = InferenceEngine(checkpoint_path=ckpt_path, tokenizer_path=vocab_path)
            engine.load_lora(lora_path)

            ids = torch.randint(0, 256, (1, 8))
            with torch.no_grad():
                expected = lora_model(ids)
                got = engine.model(ids)
            assert torch.allclose(got, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# GrimoireTransformer.merged_state_dict
# ---------------------------------------------------------------------------

class TestMergedStateDict:
    def test_no_lora_returns_normal_state_dict(self):
        model = _small_model()
        merged = model.merged_state_dict()
        sd = model.state_dict()
        assert set(merged.keys()) == set(sd.keys())
        for k in sd:
            assert torch.equal(merged[k], sd[k])

    def test_merged_keys_match_plain_model(self):
        plain = _small_model()
        plain_keys = set(plain.state_dict().keys())

        model = _small_model()
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        merged_keys = set(model.merged_state_dict().keys())

        assert merged_keys == plain_keys

    def test_merged_output_matches_lora_output(self):
        import copy
        model = _small_model()
        base_sd = copy.deepcopy(model.state_dict())

        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        for n, p in model.named_parameters():
            if "lora_B" in n:
                nn.init.normal_(p)
        model.eval()

        # Load merged state dict into a fresh plain model.
        merged_sd = model.merged_state_dict()
        plain = _small_model()
        plain.load_state_dict(merged_sd)
        plain.eval()

        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            assert torch.allclose(model(ids), plain(ids), atol=1e-5)

    def test_merged_state_dict_loadable_as_base_checkpoint(self):
        """merged_state_dict() produces a checkpoint reusable as a new base."""
        import copy
        import tempfile
        from grimoire_ai.llm.training.checkpoint import save_checkpoint, load_checkpoint

        model = _small_model()
        base_sd = copy.deepcopy(model.state_dict())
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        for n, p in model.named_parameters():
            if "lora_B" in n:
                nn.init.normal_(p)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = str(Path(tmp) / "merged.pt")
            save_checkpoint(
                path=ckpt,
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
                step=0,
                config_dict=_small_config().to_dict(),
                model_state_dict=model.merged_state_dict(),
            )
            loaded = load_checkpoint(ckpt)
            plain = _small_model()
            plain.load_state_dict(loaded["model"])  # must not raise


class TestEngineLoraDevice:
    def test_lora_params_on_correct_device_after_load(self):
        """engine.load_lora must move lora_A/lora_B to the engine's device."""
        import copy
        from grimoire_ai.llm.inference.engine import InferenceEngine
        from grimoire_ai.llm.model.lora import LoRALinear
        from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
        from grimoire_ai.llm.training.checkpoint import save_checkpoint

        config = _small_config()
        base_model = _small_model()
        base_sd = copy.deepcopy(base_model.state_dict())

        lora_model = _small_model()
        lora_model.load_state_dict(base_sd)
        lora_model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path  = str(Path(tmp) / "base.pt")
            lora_path  = str(Path(tmp) / "test.lora")
            vocab_path = str(Path(tmp) / "bpe.json")

            save_checkpoint(
                path=ckpt_path,
                model=base_model,
                optimizer=torch.optim.SGD(base_model.parameters(), lr=1e-3),
                step=0,
                config_dict=config.to_dict(),
            )
            save_lora(lora_model, rank=4, alpha=8.0,
                      targets=["q_proj", "v_proj"], path=lora_path)

            enc = BytePairEncoder()
            enc.train(["hello world " * 50], vocab_size=262)
            enc.save(vocab_path)

            engine = InferenceEngine(
                checkpoint_path=ckpt_path, tokenizer_path=vocab_path, device="cpu"
            )
            engine.load_lora(lora_path)

            for mod in engine.model.modules():
                if isinstance(mod, LoRALinear):
                    assert mod.lora_A.device.type == "cpu"
                    assert mod.lora_B.device.type == "cpu"
                    assert mod.base_weight.device.type == "cpu"
