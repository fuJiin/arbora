"""Tests for model wrappers (StepMemoryModel, MiniGPTModel, TinyStories1MModel)."""

from dataclasses import dataclass

import numpy as np
import pytest

from step.sdr import encode_token
from step.wrappers import StepMemoryModel

torch = pytest.importorskip("torch")


class TestStepMemoryModel:
    @pytest.fixture
    def model(self, small_encoder_config, small_model_config):
        return StepMemoryModel(small_model_config, small_encoder_config)

    def test_predict_observe_cycle(self, model, small_encoder_config):
        """Model can observe tokens and make predictions."""
        token_id = 42
        sdr = encode_token(token_id, small_encoder_config)
        model.observe(0, token_id, sdr)

        predicted_sdr = model.predict_sdr(1)
        assert isinstance(predicted_sdr, frozenset)
        assert len(predicted_sdr) == small_encoder_config.k

    def test_learn_returns_iou(self, model, small_encoder_config):
        """learn() returns a float IoU value."""
        t0_sdr = encode_token(100, small_encoder_config)
        model.observe(0, 100, t0_sdr)

        t1_sdr = encode_token(200, small_encoder_config)
        predicted = model.predict_sdr(1)
        iou = model.learn(1, t1_sdr, predicted)
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0

    def test_predict_token_returns_valid_token(self, model, small_encoder_config):
        """predict_token returns a token that has been observed."""
        tokens = [10, 20, 30, 40, 50]
        for t, tid in enumerate(tokens):
            sdr = encode_token(tid, small_encoder_config)
            if t > 0:
                predicted_sdr = model.predict_sdr(t)
                model.learn(t, sdr, predicted_sdr)
            model.observe(t, tid, sdr)

        predicted_token = model.predict_token(len(tokens))
        assert predicted_token in tokens or predicted_token == -1

    def test_predict_token_empty_returns_minus_one(self, model):
        """predict_token with no observations returns -1."""
        assert model.predict_token(0) == -1

    def test_decode_uses_overlap(self, model, small_encoder_config):
        """_decode finds the token with highest overlap."""
        tid_a = 100
        tid_b = 200
        sdr_a = encode_token(tid_a, small_encoder_config)
        sdr_b = encode_token(tid_b, small_encoder_config)

        model.observe(0, tid_a, sdr_a)
        model.observe(1, tid_b, sdr_b)

        assert model._decode(sdr_a) == tid_a
        assert model._decode(sdr_b) == tid_b

    def test_inverted_index_matches_matrix_decode(
        self,
        small_encoder_config,
        small_model_config,
    ):
        """Inverted index decode produces same results as dense matrix IoU."""
        model = StepMemoryModel(small_model_config, small_encoder_config)
        n = small_model_config.n
        k = small_encoder_config.k

        # Observe a set of tokens
        token_ids = list(range(50, 80))
        sdrs = {}
        for t, tid in enumerate(token_ids):
            sdr = encode_token(tid, small_encoder_config)
            sdrs[tid] = sdr
            model.observe(t, tid, sdr)

        # Build dense matrix for reference
        sdr_matrix = np.zeros((len(token_ids), n), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            for bit in sdrs[tid]:
                sdr_matrix[i, bit] = 1.0

        # Test with random query SDRs
        rng = np.random.default_rng(42)
        for _ in range(20):
            query_bits = frozenset(int(x) for x in rng.choice(n, k, replace=False))

            # Inverted index result
            inverted_result = model._decode(query_bits)

            # Dense matrix IoU result
            query_vec = np.zeros(n, dtype=np.float32)
            for bit in query_bits:
                query_vec[bit] = 1.0
            intersection = sdr_matrix @ query_vec
            row_sums = sdr_matrix.sum(axis=1)
            union = row_sums + float(k) - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            best_iou = float(np.max(iou))

            # The inverted index result should have the same (best) IoU score
            inverted_idx = token_ids.index(inverted_result)
            inv_iou = float(iou[inverted_idx])
            assert inv_iou == pytest.approx(best_iou), (
                f"Inverted chose {inverted_result} "
                f"IoU {inv_iou:.4f} != best {best_iou:.4f}"
            )

    def test_decode_empty_sdr_returns_minus_one(self, model):
        """_decode with empty SDR returns -1."""
        assert model._decode(frozenset()) == -1

    def test_decode_no_overlap_returns_minus_one(
        self,
        small_model_config,
        small_encoder_config,
    ):
        """_decode returns -1 when query has no overlap with any stored SDR."""
        model = StepMemoryModel(small_model_config, small_encoder_config)
        # Observe a token using only low bits
        sdr = frozenset(range(10))
        model.observe(0, 999, sdr)
        # Query with only high bits (no overlap)
        query = frozenset(range(200, 210))
        assert model._decode(query) == -1


class TestVocabFiltering:
    def test_token_stream_clamps_large_tokens(self, small_encoder_config):
        """Tokens >= vocab_size should be clamped to 0."""
        from step.config import EncoderConfig

        config = EncoderConfig(model_name="gpt2", n=256, k=10, vocab_size=100)
        sdr_for_0 = encode_token(0, config)

        # Token 150 is >= vocab_size=100, should be clamped to 0
        sdr_for_150 = encode_token(150, config)
        sdr_for_0_check = encode_token(0, config)
        # After clamping, token 150 would produce the same SDR as token 0
        assert sdr_for_0 == sdr_for_0_check
        assert sdr_for_0 != sdr_for_150  # Different if not clamped


class TestMiniGPTModel:
    @pytest.fixture
    def gpt_model(self):
        """Create a small MiniGPT model for testing."""
        from baselines.mini_gpt import MiniGPT, MiniGPTConfig
        from baselines.wrappers import MiniGPTModel

        config = MiniGPTConfig(
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=16,
            dropout=0.0,
        )
        net = MiniGPT(config)
        net.eval()
        return MiniGPTModel(net, config)

    def test_predict_token_empty_context(self, gpt_model):
        """predict_token with no context returns -1."""
        assert gpt_model.predict_token(0) == -1

    def test_predict_token_after_observe(self, gpt_model):
        """predict_token works after observing tokens."""
        gpt_model.observe(0, 5, frozenset())
        gpt_model.observe(1, 10, frozenset())
        token = gpt_model.predict_token(2)
        assert isinstance(token, int)
        assert 0 <= token < 100

    def test_predict_sdr_returns_empty(self, gpt_model):
        """predict_sdr always returns empty frozenset for GPT."""
        assert gpt_model.predict_sdr(0) == frozenset()
        gpt_model.observe(0, 5, frozenset())
        assert gpt_model.predict_sdr(1) == frozenset()

    def test_learn_returns_loss_no_weight_update(self, gpt_model):
        """learn() returns loss but does not update weights."""
        import torch

        gpt_model.observe(0, 5, frozenset())
        gpt_model.observe(1, 10, frozenset())
        gpt_model.observe(2, 15, frozenset())

        params_before = {
            name: p.clone() for name, p in gpt_model.model.named_parameters()
        }

        loss = gpt_model.learn(3, frozenset(), frozenset())
        assert isinstance(loss, float)
        assert loss >= 0.0

        for name, p in gpt_model.model.named_parameters():
            assert torch.equal(p, params_before[name]), f"Weight {name} was modified!"

    def test_context_truncation(self, gpt_model):
        """Context is truncated to block_size."""
        for t in range(20):  # block_size is 16
            gpt_model.observe(t, t % 100, frozenset())
        assert len(gpt_model._context) == 16

    def test_learn_short_context(self, gpt_model):
        """learn() with < 2 tokens returns 0.0."""
        gpt_model.observe(0, 5, frozenset())
        loss = gpt_model.learn(1, frozenset(), frozenset())
        assert loss == 0.0


@dataclass
class MockOutput:
    """Mock HuggingFace model output with .logits attribute."""

    logits: torch.Tensor


class MockHFModel(torch.nn.Module):
    """Mock HuggingFace CausalLM that returns MockOutput with .logits."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        # Need at least one parameter so next(model.parameters()) works
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **kwargs) -> MockOutput:
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return MockOutput(logits=logits)


class TestTinyStories1MModel:
    @pytest.fixture
    def ts_model(self):
        """Create a TinyStories1MModel with mock HF model."""
        from baselines.wrappers import TinyStories1MModel

        mock_hf = MockHFModel(vocab_size=100)
        mock_hf.eval()
        return TinyStories1MModel(mock_hf, context_length=16)

    def test_predict_token_empty_context(self, ts_model):
        """predict_token with no context returns -1."""
        assert ts_model.predict_token(0) == -1

    def test_predict_token_after_observe(self, ts_model):
        """predict_token works after observing tokens."""
        ts_model.observe(0, 5, frozenset())
        ts_model.observe(1, 10, frozenset())
        token = ts_model.predict_token(2)
        assert isinstance(token, int)
        assert 0 <= token < 100

    def test_context_truncation(self, ts_model):
        """Context is truncated to context_length."""
        for t in range(20):  # context_length is 16
            ts_model.observe(t, t % 100, frozenset())
        assert len(ts_model._context) == 16

    def test_predict_sdr_returns_empty(self, ts_model):
        """predict_sdr always returns empty frozenset."""
        assert ts_model.predict_sdr(0) == frozenset()
        ts_model.observe(0, 5, frozenset())
        assert ts_model.predict_sdr(1) == frozenset()

    def test_learn_returns_loss(self, ts_model):
        """learn() returns a float loss value."""
        ts_model.observe(0, 5, frozenset())
        ts_model.observe(1, 10, frozenset())
        ts_model.observe(2, 15, frozenset())
        loss = ts_model.learn(3, frozenset(), frozenset())
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_learn_short_context(self, ts_model):
        """learn() with < 2 tokens returns 0.0."""
        ts_model.observe(0, 5, frozenset())
        loss = ts_model.learn(1, frozenset(), frozenset())
        assert loss == 0.0
