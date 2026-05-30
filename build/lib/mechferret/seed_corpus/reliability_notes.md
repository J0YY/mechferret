# Reliability Notes for Long Horizon Research Agents

Long-horizon agents fail when they lose track of evidence provenance, over-trust
early search results, or continue searching without a stopping rule. The control
loop should keep a durable evidence ledger and a critic that measures source
diversity, citation density, coverage, and contradiction pressure.

Memory helps across runs, but memory records can become stale. A reliable agent
uses memory as context, then validates important claims against current sources.
Prior-run memory should be labeled separately from external evidence so users can
see when a conclusion depends on recalled information rather than fresh sources.

The system should not hide failures. It should produce gaps such as "source
diversity is low" or "contradictory claims require adjudication" so the next
agent round has a concrete target.

