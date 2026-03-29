---
name: feedback_corticostriatal_l5
description: Corticostriatal projections should come from L5, not L2/3. Accepted from output_port for now since S1 has no L5.
type: feedback
---

Corticostriatal projections (cortex → BG striatum) biologically originate from L5, not L2/3. IT-type L5 → D1 (Go), PT-type L5 → D2 (NoGo).

**Why:** When S2/S3 (which have L5) connect to BG, they should use L5 specifically, not output_port which defaults to L2/3.

**How to apply:** For v1 MiniGrid (S1 has no L5), S1.output_port → BG is acceptable since output_port IS L2/3. When adding multi-region BG inputs or restoring L5 on sensory regions, wire L5 specifically to BG. May need a `circuit.connect(s2.l5, bg.input_port, ...)` pattern distinct from the default output_port.
