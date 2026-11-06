"""The registered reading of the e-prop profile, computed on the node right after the recording.

The registration is sp4-direction-scoping/tier2_project_2026-11-06/REGISTRATION.md.

**What it reads:** one rank's write-out of perf_recording.py (MODE "events", recorded during the measured nest.Run
only). It sums the periods of the cycles and instructions samples by class, and writes perf_<rank>_shares.json beside
the write-out. The sum of a class's periods estimates how many of the event fell in it.

**The format:** `perf script -F tid,event,period,ip,sym,dso --no-inline`, as nest_perf/perf_samples.py of the
delivery_v1 proxy reads it: a header line per sample (thread, period, event), then its call stack, leaf first.

**Threads:** the program's OpenMP threads count, meaning those with libgomp anywhere in a stack. The shares over all
threads are written beside them, in case the 4 kB stacks never reach libgomp.

**Classes,** each sample in the first whose test its stack meets:
- gradient: any frame named ...compute_gradient, meaning the per-synapse loop and everything it calls. Its share of
  cycles is f;
- erase: any frame named ...erase_used_eprop_history;
- send: any frame of eprop_synapse's send, the e-prop synapse's own work outside the loop;
- neuron_update: any frame of eprop_iaf's or eprop_readout's update, the neurons' per-step work and their history
  writes, unless inlined elsewhere;
- mpi: a frame in an MPI or fabric library, or one named MPI_...;
- other.

**Within gradient,** the part is set by the leaf frame:
- optimizer: WeightOptimizer...;
- libm: a libm frame, or pow, exp or log;
- lookup: get_eprop_history, lower_bound or require_eprop_history_entry;
- loop: compute_gradient's own instructions;
- other.

**It never raises:** a failure is written into the JSON.
"""
import gzip
import json
import os
import re
from collections import defaultdict

HEADER = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\S+?):?\s*$")
FRAME = re.compile(r"^\s+[0-9a-f]+\s+(.*?)\s+\((.*)\)\s*$")
MPI_LIBS = ("libmpi", "libmca", "libucp", "libuct", "libucs", "libfabric", "libopen-pal", "libpsm", "libibverbs")
CLASSES = ("gradient", "erase", "send", "neuron_update", "mpi", "other")
PARTS = ("optimizer", "libm", "lookup", "loop", "other")
EVENTS = ("cycles", "instructions")


def samples(path):
    """(thread, event, period, frames) per sample, frames as (symbol, library) from the leaf; a line that is neither a
    header nor a frame gives (None, None, 0, []), so that the share parsed can be judged."""
    current = None
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if not line.strip():
                if current:
                    yield current
                current = None
                continue
            frame = FRAME.match(line) if current else None
            if frame:
                current[3].append((frame.group(1), frame.group(2)))
                continue
            header = HEADER.match(line)
            if header and current is None:
                current = (int(header.group(1)), header.group(3).rstrip(":"), int(header.group(2)), [])
            else:
                yield (None, None, 0, [])
    if current:
        yield current


def event_name(ev):
    """cycles:u, cycles/period=1000/u and cpu/ls_dispatch.ld_dispatch,period=1000/u give cycles, cycles and
    ls_dispatch.ld_dispatch."""
    parts = ev.split("/")
    name = parts[1].split(",")[0] if len(parts) > 1 and parts[0] == "cpu" else parts[0]
    return name.split(":")[0]


def has(frames, *needles):
    return any(all(n in sym for n in needles) for sym, _ in frames)


def classify(frames):
    if has(frames, "compute_gradient"):
        return "gradient"
    if has(frames, "erase_used_eprop_history"):
        return "erase"
    if has(frames, "eprop_synapse", "send"):
        return "send"
    if has(frames, "eprop_iaf", "update") or has(frames, "eprop_readout", "update"):
        return "neuron_update"
    if any(sym.startswith("MPI_") or any(lib in dso for lib in MPI_LIBS) for sym, dso in frames):
        return "mpi"
    return "other"


def gradient_part(frames):
    sym, dso = frames[0]
    if "WeightOptimizer" in sym or "optimized_weight" in sym:
        return "optimizer"
    if "libm" in dso or re.match(r"^(__)?(pow|exp|log)", sym):
        return "libm"
    if any(k in sym for k in ("get_eprop_history", "lower_bound", "require_eprop_history_entry")):
        return "lookup"
    if "compute_gradient" in sym:
        return "loop"
    return "other"


def tally(path, only=None):
    """Per event, the summed periods by class and, within gradient, by part; the sample counts; the threads."""
    periods = {ev: defaultdict(int) for ev in EVENTS}
    parts = {ev: defaultdict(int) for ev in EVENTS}
    counts = {ev: defaultdict(int) for ev in EVENTS}
    threads, unparsed = set(), 0
    for tid, ev, p, frames in samples(path):
        if tid is None:
            unparsed += 1
            continue
        if only is not None and tid not in only:
            continue
        name = event_name(ev)
        if name not in EVENTS or not frames:
            continue
        c = classify(frames)
        periods[name][c] += p
        counts[name][c] += 1
        if c == "gradient":
            parts[name][gradient_part(frames)] += p
        threads.add(tid)
    out = {"threads": len(threads), "unparsed_lines": unparsed}
    for ev in EVENTS:
        total = sum(periods[ev].values())
        out[ev] = {"total": total, "samples": sum(counts[ev].values()),
                   "by_class": {c: periods[ev].get(c, 0) for c in CLASSES},
                   "share_by_class": {c: (periods[ev].get(c, 0) / total if total else None) for c in CLASSES},
                   "gradient_parts": {k: parts[ev].get(k, 0) for k in PARTS},
                   "samples_by_class": {c: counts[ev].get(c, 0) for c in CLASSES}}
    cyc = out["cycles"]
    out["f"] = cyc["share_by_class"]["gradient"]
    return out


def write_shares(recordings, rank):
    """perf_<rank>_shares.json from perf_<rank>_script.txt.gz, if this rank recorded. Never raises."""
    script = os.path.join(recordings, "perf_{}_script.txt.gz".format(rank))
    target = os.path.join(recordings, "perf_{}_shares.json".format(rank))
    if not os.path.exists(script):
        return
    result = {"rank": rank, "script": os.path.basename(script)}
    try:
        omp = {tid for tid, _, _, frames in samples(script)
               if tid is not None and any("libgomp" in dso for _, dso in frames)}
        result["openmp_threads"] = len(omp)
        result["all_threads"] = tally(script)
        result["openmp_only"] = tally(script, only=omp) if omp else None
        chosen = result["openmp_only"] if omp else result["all_threads"]
        result["f"] = chosen["f"]
        result["f_basis"] = "openmp_only" if omp else "all_threads"
    except Exception as err:  # noqa: BLE001
        result["error"] = repr(err)
    try:
        with open(target, "w") as f:
            json.dump(result, f, indent=1)
    except OSError:
        pass
