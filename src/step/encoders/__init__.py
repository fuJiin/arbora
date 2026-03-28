from step.encoders.charbit import CharbitEncoder
from step.encoders.onehot import OneHotCharEncoder
from step.encoders.positional import PositionalCharEncoder

__all__ = [
    "CharbitEncoder",
    "MiniGridEncoder",
    "OneHotCharEncoder",
    "PositionalCharEncoder",
]


def __getattr__(name: str):
    if name == "MiniGridEncoder":
        from step.encoders.minigrid import MiniGridEncoder

        return MiniGridEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
