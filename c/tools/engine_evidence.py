"""Helpers for reading what the engine wrote at startup.

Recognizes the two typed lines the engine prints at startup -- the
"== GLM C engine ..." banner and the following "loaded in ..." record --
and returns their fields as typed values, used by the evidence checkers
that read raw engine stdout. A line that merely looks like one of these
preambles but fails a field check is a bug worth surfacing loudly, so
parsing raises rather than silently skipping.
"""

import math
import re


class PreambleError(ValueError):
    """A line resembles an owned engine preamble but is not source-valid."""


_INT32_MAX = 2**31 - 1
_UINT_TEXT = r"(?:0|[1-9][0-9]*)"
_FIXED2_TEXT = r"(?:0|[1-9][0-9]*)\.[0-9]{2}"
IDOT_KERNELS = (
    "avx512-vnni", "avx-vnni", "avx2", "neon-i8mm", "neon", "vsx",
    "scalar",
)

_BANNER_RE = re.compile(
    r"^== GLM C engine \(glm_moe_dsa\), cache=(?P<cap>" + _UINT_TEXT +
    r") experts/layer \| compute experts@(?P<expert_bits>" + _UINT_TEXT +
    r")-bit dense@(?P<dense_bits>" + _UINT_TEXT +
    r")-bit \| idot: (?P<kernel>" + "|".join(IDOT_KERNELS) + r") ==$")
_LOADED_RE = re.compile(
    r"^loaded in (?P<load_s>" + _FIXED2_TEXT +
    r")s \| resident dense: (?P<resident_mb>" + _FIXED2_TEXT +
    r") MB \| layers=(?P<layers>" + _UINT_TEXT +
    r") experts=(?P<experts>" + _UINT_TEXT +
    r") \| MTP (?P<mtp>ACTIVE|absent|DISABLED \(multiplexed serve\)) "
    r"\(draft=(?P<draft>" + _UINT_TEXT + r")\)$")


def parse_engine_banner(line):
    """Return typed fields for the exact production "== GLM C engine" banner."""
    if not isinstance(line, str):
        raise PreambleError(f"engine banner is not text: {line!r}")
    match = _BANNER_RE.fullmatch(line)
    if not match:
        raise PreambleError(f"not an exact engine banner: {line!r}")
    cap, expert_bits, dense_bits = map(
        int, match.group("cap", "expert_bits", "dense_bits"))
    if not 1 <= cap <= _INT32_MAX:
        raise PreambleError(f"engine cache outside [1,{_INT32_MAX}]: {cap}")
    if not 1 <= expert_bits <= 16 or not 1 <= dense_bits <= 16:
        raise PreambleError(
            f"engine compute bits outside [1,16]: {expert_bits}/{dense_bits}")
    return {
        "kind": "BANNER", "cap": cap, "expert_bits": expert_bits,
        "dense_bits": dense_bits, "kernel": match.group("kernel"),
    }


def parse_engine_loaded(line):
    """Return typed fields for the exact "loaded in ..." record that follows the banner."""
    if not isinstance(line, str):
        raise PreambleError(f"engine load record is not text: {line!r}")
    match = _LOADED_RE.fullmatch(line)
    if not match:
        raise PreambleError(f"not an exact engine load record: {line!r}")
    load_s = float(match.group("load_s"))
    resident_mb = float(match.group("resident_mb"))
    layers, experts, draft = map(
        int, match.group("layers", "experts", "draft"))
    mtp = match.group("mtp")
    if not math.isfinite(load_s) or not math.isfinite(resident_mb):
        raise PreambleError("engine load metrics must be finite")
    if load_s < 0 or resident_mb < 0:
        raise PreambleError("engine load metrics must be nonnegative")
    if not 1 <= layers <= 128:
        raise PreambleError(f"engine layers outside [1,128]: {layers}")
    if not 1 <= experts <= 4096:
        raise PreambleError(f"engine experts outside [1,4096]: {experts}")
    if not 0 <= draft <= 63:
        raise PreambleError(f"engine draft outside [0,63]: {draft}")
    if mtp == "DISABLED (multiplexed serve)" and draft != 0:
        raise PreambleError("disabled multiplexed MTP requires draft=0")
    return {
        "kind": "LOADED", "load_s": load_s, "resident_mb": resident_mb,
        "layers": layers, "experts": experts, "mtp": mtp, "draft": draft,
    }


def parse_engine_preamble(line):
    """Dispatch to the banner/loaded parser by prefix, or return None.

    None means the line is not one of the two owned preambles at all (an
    ordinary log line); a line that starts like one of them but fails to
    parse still raises PreambleError rather than being treated as unowned.
    """
    if not isinstance(line, str):
        raise PreambleError(f"engine preamble is not text: {line!r}")
    if line.startswith("== GLM C engine"):
        return parse_engine_banner(line)
    if line.startswith("loaded in"):
        return parse_engine_loaded(line)
    return None
