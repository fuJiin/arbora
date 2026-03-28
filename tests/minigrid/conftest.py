import pytest

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")
pytest.importorskip("minigrid", reason="minigrid not installed")
