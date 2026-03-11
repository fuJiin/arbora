"""Tests for torch-dependent model wrappers (MiniGPTModel, TinyStories1MModel)."""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")


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
