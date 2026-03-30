import numpy as np
import pytest

from arbor.encoders.minigrid import MiniGridEncoder
from arbor.environment.minigrid import MiniGridObs


@pytest.fixture()
def encoder():
    return MiniGridEncoder()


@pytest.fixture()
def sample_obs():
    """Minimal 7x7x3 observation (empty room, agent facing right)."""
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    # Fill with "empty" cells (object_type=1, color=0, state=0)
    image[:, :, 0] = 1
    # Walls around edges (object_type=2, color=5=grey)
    image[0, :, 0] = 2
    image[0, :, 1] = 5
    image[-1, :, 0] = 2
    image[-1, :, 1] = 5
    # Goal at (5, 5) (object_type=8, color=1=green)
    image[5, 5, 0] = 8
    image[5, 5, 1] = 1
    return MiniGridObs(image=image, direction=0)


class TestMiniGridEncoder:
    def test_output_shape(self, encoder, sample_obs):
        """Encoding should be a flat 984-element boolean vector."""
        result = encoder.encode(sample_obs)
        assert result.shape == (984,)
        assert result.dtype == np.bool_

    def test_input_dim_property(self, encoder):
        assert encoder.input_dim == 984

    def test_encoding_width_property(self, encoder):
        assert encoder.encoding_width == 20

    def test_sparsity(self, encoder, sample_obs):
        """Each cell contributes 3 active bits + 1 direction = 148 total."""
        result = encoder.encode(sample_obs)
        active = int(result.sum())
        # 49 cells * 3 channels + 1 direction = 148
        assert active == 148

    def test_direction_one_hot(self, encoder):
        """Direction bits (last 4) should have exactly one active."""
        image = np.ones((7, 7, 3), dtype=np.uint8)
        for direction in range(4):
            obs = MiniGridObs(image=image, direction=direction)
            result = encoder.encode(obs)
            dir_bits = result[-4:]
            assert dir_bits.sum() == 1
            assert dir_bits[direction]

    def test_deterministic(self, encoder, sample_obs):
        """Same observation should produce identical encoding."""
        a = encoder.encode(sample_obs)
        b = encoder.encode(sample_obs)
        np.testing.assert_array_equal(a, b)

    def test_unseen_cells(self, encoder):
        """Unseen cells (all zeros) should still encode object_type=0."""
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        obs = MiniGridObs(image=image, direction=0)
        result = encoder.encode(obs)
        # Cell (0,0): object_type=0 → bit 0 active, color=0 → bit 11, state=0 → bit 17
        assert result[0]  # object_type 0
        assert result[11]  # color 0
        assert result[17]  # state 0

    def test_reset_is_noop(self, encoder):
        """Reset should not raise."""
        encoder.reset()

    def test_different_obs_different_encoding(self, encoder):
        """Different observations should produce different encodings."""
        img1 = np.zeros((7, 7, 3), dtype=np.uint8)
        img2 = np.zeros((7, 7, 3), dtype=np.uint8)
        img2[3, 3, 0] = 5  # key at center
        obs1 = MiniGridObs(image=img1, direction=0)
        obs2 = MiniGridObs(image=img2, direction=0)
        r1 = encoder.encode(obs1)
        r2 = encoder.encode(obs2)
        assert not np.array_equal(r1, r2)
