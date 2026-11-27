"""A stand-in for `soxr`, so the harness runs in an install that leaves that package out.

Pipecat's core install requires two LGPL packages, `soxr` and `num2words`. The harness uses neither:

- `num2words` is imported only by Pipecat's text transforms for speech synthesis, which the harness never
  imports. Leaving it out needs nothing here.
- `soxr` is imported at the top of two Pipecat audio modules that the realtime service imports, so without it
  the import fails. It is only ever CALLED to change a sample rate. The harness runs every path at 24 kHz, the
  rate the model speaks (device.SAMPLE_RATE), and a WebRTC device's audio is resampled by the transport's own
  library, so nothing calls it.

Where `soxr` is not installed, `install()` puts a module of that name in its place. Importing it works. Calling
anything in it raises, naming the reason, so audio at another rate fails where it happens instead of passing on
unresampled. Where `soxr` is installed, nothing is changed.
"""
from __future__ import annotations

import importlib.util
import sys
import types

NO_RESAMPLER = ("soxr.{name}: this install has no resampler (the LGPL `soxr` package is left out). "
                "Realtime Harness runs at 24 kHz only; install `soxr` to resample other rates.")


def _stand_in() -> types.ModuleType:
    mod = types.ModuleType("soxr", "Stands in for the soxr package, which this install leaves out.")
    mod.STAND_IN = True

    def __getattr__(name: str):
        if name.startswith("__"):          # what the import system and inspection ask a module for
            raise AttributeError(name)
        raise RuntimeError(NO_RESAMPLER.format(name=name))
    mod.__getattr__ = __getattr__
    return mod


def install() -> bool:
    """Put the stand-in in place when `soxr` cannot be imported. True when it did."""
    if sys.modules.get("soxr") is not None:
        return False
    if "soxr" not in sys.modules and importlib.util.find_spec("soxr") is not None:
        return False
    sys.modules["soxr"] = _stand_in()
    return True
