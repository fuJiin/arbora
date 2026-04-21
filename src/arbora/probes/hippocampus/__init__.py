"""Hippocampus-specific probes.

Convenience subpackage for probes that operate on `HippocampalRegion`.
The cortex / generic probes stay flat at the parent level (`core.py`,
`representation.py`, etc.); if we add another domain-specific probe
family in the future, a matching subpackage mirrors that domain.
"""

from arbora.probes.hippocampus.probe import HippocampalProbe
from arbora.probes.hippocampus.retention import RetentionTracker

__all__ = ["HippocampalProbe", "RetentionTracker"]
