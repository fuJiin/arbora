"""Hippocampal module — episodic memory and pattern completion.

See `arbora-notes/.agents/HIPPOCAMPUS.md` for v1 design decisions.
"""

from arbora.hippocampus.ca1 import CA1
from arbora.hippocampus.ca3 import CA3AttractorNetwork
from arbora.hippocampus.dentate_gyrus import DentateGyrus
from arbora.hippocampus.entorhinal import EntorhinalLayer

__all__ = [
    "CA1",
    "CA3AttractorNetwork",
    "DentateGyrus",
    "EntorhinalLayer",
]
