"""Realtime Harness: a harness for realtime speech models, served over UHP.

A realtime model already runs its own conversation loop. The harness owns what the model cannot: the
configuration against the caller, every tool call, the record of the call as a UHP session, and the
credential and the meter (docs/design.md §2).
"""
__version__ = "0.1.0.dev0"
