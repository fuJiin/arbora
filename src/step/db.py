"""SQLite-backed STEP model.

Uses SQL aggregation for predict/learn and lazy weight decay via
POW(decay, dt) at read time. Gives us persistence, replay, and
queryability for free.
"""

import json
import sqlite3
from pathlib import Path

from step.config import EncoderConfig, ModelConfig


class StepModel:
    def __init__(
        self,
        db_path: Path | str,
        config: ModelConfig,
        encoder_config: EncoderConfig,
        commit_interval: int = 100,
    ):
        self.config = config
        self.encoder_config = encoder_config
        self.db_path = Path(db_path)
        self._commit_interval = commit_interval
        self._ops_since_commit = 0

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._create_schema()
        self._save_metadata()

    def _create_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS synapses (
                src INTEGER,
                dst INTEGER,
                w REAL,
                last_updated_t INTEGER DEFAULT 0,
                PRIMARY KEY (src, dst)
            );
            CREATE INDEX IF NOT EXISTS idx_src ON synapses(src);

            CREATE TABLE IF NOT EXISTS sdr_history (
                timestamp_t INTEGER PRIMARY KEY,
                token_id INTEGER
            );

            CREATE TABLE IF NOT EXISTS sdr_definitions (
                token_id INTEGER,
                bit_index INTEGER,
                token_str TEXT,
                PRIMARY KEY (token_id, bit_index)
            );
            CREATE INDEX IF NOT EXISTS idx_bit_to_token ON sdr_definitions(bit_index);

            CREATE TABLE IF NOT EXISTS metrics (
                step INTEGER PRIMARY KEY,
                iou REAL,
                rolling_iou REAL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)

    def _save_metadata(self) -> None:
        from dataclasses import asdict

        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("model_config", json.dumps(asdict(self.config))),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("encoder_config", json.dumps(asdict(self.encoder_config))),
        )
        self.conn.commit()

    def predict_token(self, t: int) -> int:
        """Predict the next token ID."""
        sdr = self.predict_sdr(t)
        return self.decode(sdr)

    def predict_sdr(self, t: int) -> frozenset[int]:
        """Predict the next SDR using SQL aggregation with lazy weight decay."""
        window = self.config.eligibility_window
        k = self.config.k
        weight_decay = self.config.weight_decay

        rows = self.conn.execute(
            """
            WITH recent_bits AS (
                SELECT d.bit_index,
                       1.0 - ((:current_t - h.timestamp_t) * 1.0 / :window) AS strength
                FROM sdr_history h
                JOIN sdr_definitions d ON h.token_id = d.token_id
                WHERE h.timestamp_t > (:current_t - :window)
                  AND h.timestamp_t < :current_t
            )
            SELECT s.dst, SUM(
                s.w
                * POWER(:weight_decay, :current_t - s.last_updated_t)
                * rb.strength
            ) AS vote
            FROM synapses s
            JOIN recent_bits rb ON s.src = rb.bit_index
            GROUP BY s.dst
            ORDER BY vote DESC
            LIMIT :k
            """,
            {
                "current_t": t,
                "window": window,
                "weight_decay": weight_decay,
                "k": k,
            },
        ).fetchall()

        if len(rows) < k:
            # Not enough synapse data yet; fill with arbitrary indices
            predicted = {r[0] for r in rows}
            for i in range(self.config.n):
                if len(predicted) >= k:
                    break
                predicted.add(i)
            return frozenset(predicted)

        return frozenset(r[0] for r in rows)

    def learn(
        self, t: int, actual_sdr: frozenset[int], predicted_sdr: frozenset[int]
    ) -> float:
        """Update weights via SQL. Returns IoU."""
        overlap = len(actual_sdr & predicted_sdr)
        iou = overlap / self.config.k
        actual_eta = self.config.max_lr * (1.0 - iou)
        window = self.config.eligibility_window
        weight_decay = self.config.weight_decay

        if actual_eta == 0.0:
            return iou

        # Compute recent bits with strengths
        recent_bits = self.conn.execute(
            """
            SELECT d.bit_index,
                   1.0 - ((:current_t - h.timestamp_t) * 1.0 / :window) AS strength
            FROM sdr_history h
            JOIN sdr_definitions d ON h.token_id = d.token_id
            WHERE h.timestamp_t > (:current_t - :window)
              AND h.timestamp_t < :current_t
            """,
            {"current_t": t, "window": window},
        ).fetchall()

        if not recent_bits:
            return iou

        # Aggregate deltas per unique (src, dst) to minimize upserts.
        # A src_bit appears once per history timestep with different strengths;
        # summing first collapses O(window * k) rows to O(unique_src * k).
        aggregated: dict[tuple[int, int], float] = {}

        # Reinforcement: for each (src_bit, strength) x each actual_bit
        for src_bit, strength in recent_bits:
            delta = actual_eta * strength
            for dst_bit in actual_sdr:
                key = (src_bit, dst_bit)
                aggregated[key] = aggregated.get(key, 0.0) + delta

        # Penalization: for each (src_bit, strength) x each false_positive_bit
        false_positives = predicted_sdr - actual_sdr
        if false_positives and self.config.penalty_factor > 0:
            for src_bit, strength in recent_bits:
                delta = -actual_eta * self.config.penalty_factor * strength
                for dst_bit in false_positives:
                    key = (src_bit, dst_bit)
                    aggregated[key] = aggregated.get(key, 0.0) + delta

        if aggregated:
            self.conn.executemany(
                """
                INSERT INTO synapses (src, dst, w, last_updated_t)
                VALUES (:src, :dst, :delta, :t)
                ON CONFLICT(src, dst) DO UPDATE SET
                    w = synapses.w
                        * POWER(:decay, :t - synapses.last_updated_t)
                        + :delta,
                    last_updated_t = :t
                """,
                [
                    {"src": s, "dst": d, "delta": delta, "t": t, "decay": weight_decay}
                    for (s, d), delta in aggregated.items()
                ],
            )

        self._maybe_commit()
        return iou

    def observe(self, t: int, token_id: int, sdr: frozenset[int]) -> None:
        """Record a token observation and cache its SDR definition."""
        # Cache SDR definition if new token
        self.conn.executemany(
            "INSERT OR IGNORE INTO sdr_definitions (token_id, bit_index, token_str) "
            "VALUES (?, ?, ?)",
            [(token_id, bit, "") for bit in sdr],
        )

        # Update history
        self.conn.execute(
            "INSERT OR REPLACE INTO sdr_history (timestamp_t, token_id) VALUES (?, ?)",
            (t, token_id),
        )

        # Prune old history
        self.conn.execute(
            "DELETE FROM sdr_history WHERE timestamp_t <= ?",
            (t - self.config.eligibility_window,),
        )
        self._maybe_commit()

    def _maybe_commit(self) -> None:
        """Commit periodically to batch disk I/O."""
        self._ops_since_commit += 1
        if self._ops_since_commit >= self._commit_interval:
            self.conn.commit()
            self._ops_since_commit = 0

    def flush(self) -> None:
        """Force commit any pending changes."""
        self.conn.commit()
        self._ops_since_commit = 0

    def decode(self, sdr: frozenset[int]) -> int:
        """Decode an SDR to the best-matching token_id via inverted index."""
        if not sdr:
            return -1

        placeholders = ",".join("?" for _ in sdr)
        row = self.conn.execute(
            f"""
            SELECT token_id, COUNT(*) AS overlap
            FROM sdr_definitions
            WHERE bit_index IN ({placeholders})
            GROUP BY token_id
            ORDER BY overlap DESC
            LIMIT 1
            """,
            list(sdr),
        ).fetchone()

        return row[0] if row else -1

    def log_metrics(self, step: int, iou: float, rolling_iou: float) -> None:
        """Log metrics to the metrics table."""
        self.conn.execute(
            "INSERT OR REPLACE INTO metrics (step, iou, rolling_iou) VALUES (?, ?, ?)",
            (step, iou, rolling_iou),
        )
        self._maybe_commit()

    def close(self) -> None:
        self.flush()
        self.conn.close()

    def __enter__(self) -> "StepModel":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
