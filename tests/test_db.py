from pathlib import Path

import pytest

from step.config import EncoderConfig, ModelConfig
from step.db import StepModel


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        n=64,
        k=4,
        max_lr=0.5,
        weight_decay=0.999,
        penalty_factor=0.5,
        eligibility_window=10,
    )


@pytest.fixture
def encoder_config() -> EncoderConfig:
    return EncoderConfig(n=64, k=4)


@pytest.fixture
def model(
    db_path: Path, config: ModelConfig, encoder_config: EncoderConfig
) -> StepModel:
    m = StepModel(db_path, config, encoder_config)
    yield m
    m.close()


class TestSchema:
    def test_creates_tables(self, model: StepModel):
        tables = model.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "synapses" in table_names
        assert "sdr_history" in table_names
        assert "sdr_definitions" in table_names
        assert "metrics" in table_names
        assert "metadata" in table_names

    def test_stores_metadata(self, model: StepModel):
        row = model.conn.execute(
            "SELECT value FROM metadata WHERE key = 'model_config'"
        ).fetchone()
        assert row is not None
        assert '"n": 64' in row[0]


class TestObserve:
    def test_records_history(self, model: StepModel):
        sdr = frozenset({0, 1, 2, 3})
        model.observe(0, token_id=42, sdr=sdr)

        row = model.conn.execute(
            "SELECT token_id FROM sdr_history WHERE timestamp_t = 0"
        ).fetchone()
        assert row[0] == 42

    def test_caches_sdr_definition(self, model: StepModel):
        sdr = frozenset({10, 20, 30, 40})
        model.observe(0, token_id=7, sdr=sdr)

        bits = model.conn.execute(
            "SELECT bit_index FROM sdr_definitions "
            "WHERE token_id = 7 ORDER BY bit_index"
        ).fetchall()
        assert [b[0] for b in bits] == [10, 20, 30, 40]

    def test_prunes_old_history(self, model: StepModel):
        for t in range(20):
            model.observe(t, token_id=t, sdr=frozenset({t % 64}))

        rows = model.conn.execute("SELECT COUNT(*) FROM sdr_history").fetchone()
        # Window is 10, so at t=19 we keep t=10..19
        assert rows[0] == 10


class TestPredict:
    def test_returns_k_indices(self, model: StepModel):
        sdr = frozenset({0, 1, 2, 3})
        model.observe(0, token_id=1, sdr=sdr)
        pred = model.predict_sdr(t=1)
        assert len(pred) == 4

    def test_all_in_range(self, model: StepModel):
        sdr = frozenset({0, 1, 2, 3})
        model.observe(0, token_id=1, sdr=sdr)
        pred = model.predict_sdr(t=1)
        assert all(0 <= idx < 64 for idx in pred)

    def test_empty_returns_k(self, model: StepModel):
        pred = model.predict_sdr(t=0)
        assert len(pred) == 4


class TestLearn:
    def test_returns_iou(self, model: StepModel):
        sdr0 = frozenset({0, 1, 2, 3})
        model.observe(0, token_id=1, sdr=sdr0)

        actual = frozenset({0, 1, 2, 3})
        predicted = frozenset({0, 1, 4, 5})
        iou = model.learn(t=1, actual_sdr=actual, predicted_sdr=predicted)
        assert iou == 0.5

    def test_creates_synapses(self, model: StepModel):
        sdr0 = frozenset({10, 11, 12, 13})
        model.observe(0, token_id=1, sdr=sdr0)

        actual = frozenset({0, 1, 2, 3})
        predicted = frozenset({0, 1, 4, 5})
        model.learn(t=1, actual_sdr=actual, predicted_sdr=predicted)

        count = model.conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
        assert count > 0

    def test_reinforces_correct_bits(self, model: StepModel):
        sdr0 = frozenset({10, 11, 12, 13})
        model.observe(0, token_id=1, sdr=sdr0)

        actual = frozenset({0, 1, 2, 3})
        predicted = frozenset({0, 1, 4, 5})
        model.learn(t=1, actual_sdr=actual, predicted_sdr=predicted)

        row = model.conn.execute(
            "SELECT w FROM synapses WHERE src = 10 AND dst = 0"
        ).fetchone()
        assert row is not None
        assert row[0] > 0

    def test_penalizes_false_positives(self, model: StepModel):
        sdr0 = frozenset({10, 11, 12, 13})
        model.observe(0, token_id=1, sdr=sdr0)

        actual = frozenset({0, 1, 2, 3})
        predicted = frozenset({0, 1, 4, 5})
        model.learn(t=1, actual_sdr=actual, predicted_sdr=predicted)

        # False positive: bit 4 was predicted but not actual
        row = model.conn.execute(
            "SELECT w FROM synapses WHERE src = 10 AND dst = 4"
        ).fetchone()
        assert row is not None
        assert row[0] < 0


class TestLazyDecay:
    def test_decay_applied_at_read_time(
        self, db_path: Path, encoder_config: EncoderConfig
    ):
        """Weights decay lazily via POW(decay, dt) at predict time."""
        config = ModelConfig(
            n=64,
            k=4,
            max_lr=0.5,
            weight_decay=0.5,
            penalty_factor=0.5,
            eligibility_window=20,
        )
        with StepModel(db_path, config, encoder_config) as m:
            sdr0 = frozenset({0, 1, 2, 3})
            m.observe(0, token_id=1, sdr=sdr0)

            actual = frozenset({10, 11, 12, 13})
            predicted = frozenset({20, 21, 22, 23})
            m.learn(t=1, actual_sdr=actual, predicted_sdr=predicted)

            row = m.conn.execute(
                "SELECT w, last_updated_t FROM synapses WHERE src = 0 AND dst = 10"
            ).fetchone()
            assert row is not None
            raw_w = row[0]
            assert row[1] == 1

            # Raw weight unchanged (lazy decay)
            row2 = m.conn.execute(
                "SELECT w FROM synapses WHERE src = 0 AND dst = 10"
            ).fetchone()
            assert row2[0] == raw_w


class TestDecode:
    def test_decode_known_token(self, model: StepModel):
        sdr = frozenset({5, 10, 15, 20})
        model.observe(0, token_id=42, sdr=sdr)

        decoded = model.decode(sdr)
        assert decoded == 42

    def test_decode_partial_match(self, model: StepModel):
        sdr1 = frozenset({0, 1, 2, 3})
        sdr2 = frozenset({2, 3, 4, 5})
        model.observe(0, token_id=10, sdr=sdr1)
        model.observe(1, token_id=20, sdr=sdr2)

        # Query with bits that overlap more with sdr1
        query = frozenset({0, 1, 2, 3})
        assert model.decode(query) == 10

    def test_decode_empty_sdr(self, model: StepModel):
        assert model.decode(frozenset()) == -1


class TestPredictToken:
    def test_returns_int(self, model: StepModel):
        sdr = frozenset({0, 1, 2, 3})
        model.observe(0, token_id=1, sdr=sdr)
        token = model.predict_token(t=1)
        assert isinstance(token, int)


class TestRoundTrip:
    def test_predict_learn_observe_cycle(self, model: StepModel):
        sdrs = [frozenset({i, i + 1, i + 2, i + 3}) for i in range(0, 40, 4)]
        ious = []

        for t, sdr in enumerate(sdrs):
            if t > 0:
                pred = model.predict_sdr(t)
                iou = model.learn(t, sdr, pred)
                ious.append(iou)
            model.observe(t, token_id=t, sdr=sdr)

        assert len(ious) == len(sdrs) - 1
        assert all(0.0 <= x <= 1.0 for x in ious)


class TestPersistence:
    def test_checkpoint_resume(
        self,
        db_path: Path,
        config: ModelConfig,
        encoder_config: EncoderConfig,
    ):
        """Model state persists across close/reopen."""
        sdr = frozenset({0, 1, 2, 3})

        # First session
        with StepModel(db_path, config, encoder_config) as m:
            m.observe(0, token_id=42, sdr=sdr)
            actual = frozenset({10, 11, 12, 13})
            predicted = frozenset({10, 11, 14, 15})
            m.learn(t=1, actual_sdr=actual, predicted_sdr=predicted)

        # Second session - data should persist
        with StepModel(db_path, config, encoder_config) as m:
            count = m.conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
            assert count > 0

            history = m.conn.execute("SELECT COUNT(*) FROM sdr_history").fetchone()[0]
            assert history > 0

            decoded = m.decode(sdr)
            assert decoded == 42


class TestMetrics:
    def test_log_metrics(self, model: StepModel):
        model.log_metrics(step=1, iou=0.5, rolling_iou=0.45)
        model.log_metrics(step=2, iou=0.6, rolling_iou=0.55)

        rows = model.conn.execute(
            "SELECT step, iou, rolling_iou FROM metrics ORDER BY step"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, 0.5, 0.45)
        assert rows[1] == (2, 0.6, 0.55)
