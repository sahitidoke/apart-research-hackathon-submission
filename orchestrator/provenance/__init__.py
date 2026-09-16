"""Answer-provenance auditing for multi-agent MuSiQue collectives.

A planner and several workers answer one MuSiQue question over an explicit,
append-only message bus. Every model input is a pure function of that bus, so
an episode can be replayed with individual messages ablated, paraphrased or
swapped, and the final answer's sensitivity to each becomes measurable without
ever reading a chain of thought.
"""
