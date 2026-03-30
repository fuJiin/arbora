from arbor.encoders.charbit import CharbitEncoder
from arbor.encoders.onehot import OneHotCharEncoder
from arbor.encoders.positional import PositionalCharEncoder

__all__ = [
    "CharbitEncoder",
    "MiniGridEncoder",
    "OneHotCharEncoder",
    "PositionalCharEncoder",
]


def __getattr__(name: str):
    if name == "MiniGridEncoder":
        from arbor.encoders.minigrid import MiniGridEncoder

        return MiniGridEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
