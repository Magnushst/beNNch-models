"""perf record on ranks 0 and 1 for the measured simulation only.

Copied for the e-prop profile from beNNch-models branch save-cycle-log-perf (e10d03d),
multi-area-model/multiarea_model/perf_recording.py. Two things changed (Magnus Stromme, 6 Nov 2026):
- MODE is set to "events" below, so the whole measured simulation is sampled, with call stacks;
- ROLES keeps cycles and instructions only, at periods re-derived for this workload (see ROLES).

The recording attaches to the rank's own process with its events off, switches them on just before nest.Run(T) and
off just after, through perf's control fifos, then is written out on the node by perf script (each sample's thread,
event, period and call stack), compressed, into the run's recordings folder. The raw recording stays in
/work/scratch/$USER/nest_perf/<job> (the run's folder if that does not exist). A rank without perf, or whose perf
fails, records nothing and simulates as usual; every step is logged in perf_<rank>.json beside the write-out.
The options and events were tried on a Hambach Zen 3 node before this run (Magnus Stromme, 27 Oct 2026).
MODE "ibs" (Magnus Stromme, 30 Oct 2026) records instead with AMD's instruction-based sampling: one retired op in every
1,048,560 dispatched ops with its data address and source, load latency and raw registers, beside five core events
with fixed periods (cycles, instructions, retired ops, loads dispatched, retired mispredicted branches), 1 kB call
stacks, for the first WINDOW seconds of the measured simulation. Hambach's kernel refuses IBS for a process ("Invalid
event in per-thread mode"), so the rank's CPUs are recorded: every CPU any of its threads may run on (OpenMP binds the
main thread to one of them). The samples are told apart by process later. It was tried on the candidates first
(delivery_v1/candidates/IBS_2026-10-30.md).
MODE "fills" (Magnus Stromme, 31 Oct 2026) records the rank's own process for the first WINDOW seconds with five core
events and 1 kB call stacks: cycles; demand fills of the L1 cache from the L3 (or another L2 of the same core complex)
and from local memory; and what the L1 and L2 hardware prefetchers fetch from local memory. They tell where delivery's
misses are served from and how much the prefetchers add (delivery_v1/FILLS_2026-10-31.md).
On Zen 2 (JURECA's EPYC 7742; Magnus Stromme, 3 Nov 2026) the fills mode takes the demand fills as AMD's PMCx043 with
its local and remote bits together (raw r1243, from the L3 or another core complex's cache; r4843, from memory), so no
fill is missed whichever node Zen 2 counts as local on a node split into NUMA domains; and, as Zen 2 counts no L1
prefetcher fills, the L2 prefetches that hit the L3 (l2_pf_miss_l2_hit_l3, raw rff71) in place of the L1 prefetcher's,
beside those that miss it (rff72). The codes are those of perf 7.3's Zen 2 tables (PERF_CPUID=AuthenticAMD-23-31 perf
list --details: PMCx043 umasks 0x02 and 0x10 from a cache, 0x08 and 0x40 from memory; PMCx071 and PMCx072 umask 0xff).
The Zen 3 names stay first, so Hambach's recordings are unchanged. The log names the raw recording's path ("data").
MODE "counts" (Magnus Stromme, 1 Nov 2026) records no perf: a spike recorder, created after the whole network so that
no unit's id, thread or random numbers change (simulation_3.py), kept in memory, records every unit from the start of
the measured simulation; after it each rank writes its units' event counts (write_counts, event_counts_<rank>.npz in
the recordings folder). With network_gids.txt they give each population's rate in the measured second
(delivery_v1/candidates/RATES_2026-11-01.md). The model's own recording stays off, as in the benchmark.
MODE "compare" (Magnus Stromme, 4 Nov 2026) records ranks 0 to 9 for the first WINDOW seconds, each node's two ranks with
one group of five core events (COMPARE_GROUPS: ranks 0 and 1 the first, 2 and 3 the second, and so on), so that 25
events are counted on five nodes without two events sharing a counter: the work (cycles, instructions, retired ops,
loads and stores dispatched); the L1 cache's demand fills by where they come from (the L2, the L3 of the core's own
complex, another complex's cache, local memory, the other socket); the hardware prefetchers' fills (the L1's from the
L2, an L3 and memory; the L2's that hit and that miss the L3); the data TLB (L1 misses, L2 hits, L2 misses on 4 kB and
2 MB pages, page walks); and branches and the L1 cache (cycles, branches, mispredicted branches, L1 accesses, L1
misses by loads). Each is AMD Zen 3's code (PPR for family 19h model 01h; perf's amdzen3 tables), written r<umask><event>.
The same groups record delivery_v2 on the same nodes (delivery_v2/PERF8_2026-11-04.md).
write_pages records every rank's memory and its share on large pages as the measured simulation starts and ends
(pages_<rank>_start.txt, pages_<rank>_end.txt in the recordings folder; Magnus Stromme, 28 Oct 2026).
"""
import json
import os
import select
import signal
import socket
import subprocess
import time

RANKS = (0, 1)
# "events": the five events of 27 Oct with 4 kB stacks over the whole measured simulation; "ibs": the op sampling of
# 30 Oct; "fills": the fills and prefetches of 31 Oct; "counts": every unit's events in the measured second, no perf;
# "compare": five groups of five events on ranks 0 to 9, beside delivery_v2 (4 Nov). The e-prop profile uses "events".
MODE = "events"
# No sampling rate: each event has a fixed period, so the periods of its samples add up to its count. 512 kB of buffer
# per thread; 4 kB of each thread's stack per sample, for its call stack; events off until switched on.
OPTIONS = ["-m", "128", "--call-graph=dwarf,4096", "-D", "-1"]
# For each question the first name this node's perf accepts, with its period.
# E-prop profile (6 Nov 2026): only cycles, which decide the reading, and instructions, beside them. The multi-area
# model's three further events (loads, fills from memory, page walks) are left out, to keep the write-out short.
# The periods are five times the multi-area model's, re-derived from these assumptions (derived, unverified):
# - the measured 500 ms are expected to take about 50 s on one node, at 64 threads per rank and about 2.5 GHz;
# - so a rank holds about 8e12 cycles;
# - at 1.5e9 cycles per sample, that gives about 5,000 samples, the share f to about 0.4 percentage points.
# run_benchmark.py replaces ROLES from the config's perf_period; these values are the default.
ROLES = ((("cycles",), 1500000000), (("instructions",), 500000000))
FIELDS = "tid,event,period,ip,sym,dso"
# IBS mode: the op sampler, its period (2^20 - 16, the longest without IBS's extended counter) and options; the core
# events beside it, each the first name this perf accepts; the window; and the two write-outs, with call stacks and
# with each sample's own op, joined later by thread and time stamp.
IBS_EVENT = "ibs_op/cnt_ctl=1/"
IBS_OPTIONS = ["-c", "1048560", "-d", "-W", "-R", "-m", "128", "--call-graph=dwarf,1024", "-D", "-1"]
# The core events' periods give about 4,000 delivery samples each in the window: about 3.2e8 deliveries per rank in
# 10 s at the recorded run's 12 ms per cycle (job 3897783), at NEST's 525 cycles, 105 instructions, 110 ops and 50 loads
# per delivery; mispredicted branches at up to about 1 per delivery.
IBS_ROLES = ((("cycles",), 40000000), (("instructions",), 8000000), (("ex_ret_ops",), 8000000),
             (("ls_dispatch.ld_dispatch",), 4000000), (("ex_ret_brn_misp",), 40000))
# Fills mode: the five events, each the first name this perf accepts, at periods for about 4,000 delivery samples each
# in the window (1.16e9 deliveries per rank in the IBS run's 10 s, 2,915 cycles): cycles at NEST's 525 per delivery, the
# fills and prefetches at about 1 per delivery each (0.5 to 2 expected).
FILLS_ROLES = ((("cycles",), 150000000),
               (("ls_dmnd_fills_from_sys.int_cache", "r1243"), 300000),
               (("ls_dmnd_fills_from_sys.mem_io_local", "r4843"), 300000),
               (("ls_hw_pf_dc_fills.mem_io_local", "l2_pf_miss_l2_hit_l3", "rff71"), 300000),
               (("l2_pf_miss_l2_l3", "rff72"), 300000))
FILLS_OPTIONS = ["-m", "128", "--call-graph=dwarf,1024", "-D", "-1"]
# Compare mode: rank r records group r // 2, ranks 0 to 9. Periods for about 4,000 delivery samples per event in the
# window (as the fills mode's) at NEST's counts per delivery where measured (525 cycles, 105 instructions, 110 ops, 50
# loads, 1.8 fills from the L3, 1.1 from memory, 0.35 page walks), at 1 to 2 per delivery otherwise; 100,000 for what
# should be rare (another complex's cache, the other socket, 2 MB pages' walks).
COMPARE_GROUPS = (
    ("work", ((("cycles",), 150000000), (("instructions",), 30000000), (("r00c1",), 30000000), (("r0129",), 15000000),
              (("r0229",), 2000000))),
    ("demand fills", ((("r0143",), 300000), (("r0243",), 300000), (("r0443",), 100000), (("r0843",), 300000),
                      (("r5043",), 100000))),
    ("prefetches", ((("r015a",), 300000), (("r065a",), 300000), (("r485a",), 300000), (("rff71",), 300000),
                    (("rff72",), 300000))),
    ("tlb", ((("rff45",), 300000), (("r0f45",), 300000), (("r1045",), 100000), (("r4045",), 100000), (("r0346",), 100000))),
    ("branches and l1", ((("cycles",), 150000000), (("r00c2",), 5000000), (("r00c3",), 100000), (("r0040",), 15000000),
                         (("r0141",), 300000))))
WINDOW = 10.0   # seconds of the measured simulation recorded: about 1.6e6 samples, 2 GB per rank
IBS_FIELDS = "pid,tid,time,period,event,ip,sym,dso"
IBS_OPS_FIELDS = "pid,tid,time,period,event,ip,sym,dso,addr,data_src"


def zen3():
    """Whether this node's processor is AMD Zen 3 (family 25, model 1: Hambach's EPYC 7713), whose codes the compare
    mode's groups are."""
    try:
        with open("/proc/cpuinfo") as f:
            info = f.read()
    except OSError:
        return False
    first = lambda key: next((line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith(key)), None)  # noqa: E731
    return first("vendor_id") == "AuthenticAMD" and first("cpu family") == "25" and first("model\t") == "1"


def rank_cpus():
    """Every CPU any thread of this process may run on, as a list for perf -C (0-63,70)."""
    cpus = set()
    for task in os.listdir("/proc/self/task"):
        try:
            cpus |= os.sched_getaffinity(int(task))
        except OSError:
            pass
    cpus = sorted(cpus)
    runs, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            runs.append(str(start) if start == c else "{}-{}".format(start, c))
            start = None
    return ",".join(runs)


def accepted(event):
    try:
        out = subprocess.run(["perf", "stat", "-x,", "-e", event + ":u", "--", "true"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.split(",")[0].isdigit() for line in out.stderr.splitlines())


def with_period(event, period, probe):
    """The event with its period in the first form this perf accepts, or None. A raw code r<umask><event> (two hex
    digits each) is written as cpu/event=0x..,umask=0x..,period=../u."""
    forms = ["{}/period={}/u".format(event, period), "cpu/{},period={}/u".format(event, period)]
    if len(event) == 5 and event[0] == "r" and all(c in "0123456789abcdef" for c in event[1:]):
        forms.append("cpu/event=0x{},umask=0x{},period={}/u".format(event[3:], event[1:3], period))
    for form in forms:
        try:
            ok = subprocess.run(["perf", "record", "-q", "-e", form, "-o", probe, "--", "true"],
                                capture_output=True, timeout=60).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            return form
    return None


class Recording:
    def __init__(self, rank, recordings):
        self.rank, self.perf, self.log = rank, None, {"rank": rank, "host": socket.gethostname(), "pid": os.getpid(), "mode": MODE}
        self.window = None   # IBS mode: the helper process that switches the events off after WINDOW seconds
        self.out = os.path.join(recordings, "perf_{}".format(rank))
        scratch = os.path.join("/work/scratch", os.environ.get("USER", ""), "nest_perf", os.environ.get("SLURM_JOB_ID", "nojob"))
        try:
            os.makedirs(scratch, exist_ok=True)
        except OSError:
            scratch = recordings
        self.data = self.log["data"] = os.path.join(scratch, "rank_{}_perf.data".format(rank))

    def _write_log(self):
        with open(self.out + ".json", "w") as f:
            json.dump(self.log, f, indent=1)

    def _command(self, command, timeout):
        """Send enable or disable and wait for perf's answer; False if none comes or perf has ended."""
        os.write(self.ctl, (command + "\n").encode())
        for _ in range(timeout):
            ready, _, _ = select.select([self.ack], [], [], 1)
            if ready:
                return b"ack" in os.read(self.ack, 64)
            if self.perf.poll() is not None:
                return False
        return False

    def start(self):
        """Never raises: on any failure the rank simulates unrecorded, and the failure is logged."""
        recorded = self.rank < 2 * len(COMPARE_GROUPS) if MODE == "compare" else self.rank in RANKS
        if not recorded or MODE == "counts":
            return self
        try:
            return self._start()
        except Exception as err:  # noqa: BLE001
            self.log.update(started=False, error="start: {!r}".format(err))
            if self.perf is not None:
                self.perf.kill()
                self.perf.wait()
                self.perf = None
            self._write_log()
            return self

    def _start(self):
        roles = {"ibs": IBS_ROLES, "fills": FILLS_ROLES}.get(MODE, ROLES)
        if MODE == "compare":
            self.log["group"], roles = COMPARE_GROUPS[self.rank // 2]
            if not zen3():   # the groups' codes are Zen 3's: on another processor they count something else, so nothing
                self.log.update(started=False, error="compare mode records only on AMD Zen 3 (family 25, model 1)")
                self._write_log()
                return self
        names = [next((e for e in role if accepted(e)), None) for role, _ in roles]
        events = [with_period(e, period, self.data + ".probe") if e else None for e, (_, period) in zip(names, roles)]
        options = OPTIONS
        if MODE == "ibs":
            events, options = [IBS_EVENT] + events, IBS_OPTIONS
        elif MODE in ("fills", "compare"):
            options = FILLS_OPTIONS
        try:
            version = subprocess.run(["perf", "--version"], capture_output=True, text=True).stdout.strip()
        except OSError:
            version = "perf missing"
        self.log.update(events=events, options=options, affinity=sorted(os.sched_getaffinity(0)), perf_version=version)
        if None in events:
            self.log["started"] = False
            self._write_log()
            return self
        ctl, ack = self.ctl_path, self.ack_path = self.data + ".ctl", self.data + ".ack"
        for fifo in (ctl, ack):
            if os.path.exists(fifo):
                os.remove(fifo)
            os.mkfifo(fifo)
        # both ends held open here, so neither side ever reads an end of file
        self.ctl, self.ack = os.open(ctl, os.O_RDWR), os.open(ack, os.O_RDWR)
        if MODE == "ibs":
            self.log["cpus"] = rank_cpus()
            target = ["-C", self.log["cpus"]]
        else:
            target = ["-p", str(os.getpid())]
        with open(self.out + "_record.log", "w") as err:
            self.perf = subprocess.Popen(["perf", "record", *options, "-e", ",".join(events), "--control",
                                          "fifo:{},{}".format(ctl, ack), *target, "-o", self.data],
                                         stdout=err, stderr=subprocess.STDOUT)
        try:
            self.log["perf_affinity"] = sorted(os.sched_getaffinity(self.perf.pid))
        except OSError:
            self.log["perf_affinity"] = None
        self.log["started"] = self._command("enable", 300)
        self.log["enabled_at"] = time.time()
        if self.log["started"] and MODE in ("ibs", "fills", "compare"):
            # a process of its own, as nest.Run may hold Python's lock for the whole simulation: after WINDOW seconds it
            # switches the events off through the control fifo and waits for perf's answer
            self.window = subprocess.Popen(["bash", "-c", "sleep {}; echo disable > '{}'; read -t 300 answer < '{}'; echo \"$answer\"".format(
                WINDOW, self.ctl_path, self.ack_path)], stdout=subprocess.PIPE, text=True)
        if not self.log["started"]:
            self.perf.kill()
            self.perf.wait()
            self.perf = None
            self._close()
        self._write_log()
        return self

    def _close(self):
        for fd in (self.ctl, self.ack):
            try:
                os.close(fd)
            except OSError:
                pass

    def stop(self):
        """Never raises: a failure is logged."""
        if self.perf is None:
            return
        try:
            self._stop()
        except Exception as err:  # noqa: BLE001
            self.log["error"] = "stop: {!r}".format(err)
            if self.perf.poll() is None:
                self.perf.kill()
            self._write_log()

    def _stop(self):
        self.log["disabled_at"] = time.time()
        if self.window is not None and self.window.poll() is not None:
            # the window closed during the simulation: perf's events are already off
            self.log["window_answer"] = self.window.stdout.read().strip()
            self.log["disabled"] = "ack" in self.log["window_answer"]
            self.log["window_seconds"] = WINDOW
        else:
            if self.window is not None:   # the simulation ended inside the window
                self.window.kill()
                self.window.wait()
                self.log["window_seconds"] = self.log["disabled_at"] - self.log["enabled_at"]
            self.log["disabled"] = self._command("disable", 300)
        self.perf.send_signal(signal.SIGINT)
        try:
            self.log["perf_status"] = self.perf.wait(timeout=900)
        except subprocess.TimeoutExpired:
            self.perf.kill()
            self.log["perf_status"] = "killed after 900 s"
        self._close()
        start = time.time()
        with open(self.out + "_stats.txt", "w") as f:
            subprocess.run(["perf", "report", "-i", self.data, "--stats"], stdout=f, stderr=subprocess.STDOUT)
        fields = IBS_FIELDS if MODE == "ibs" else FIELDS
        script = "perf script -i '{}' -F {} --no-inline 2> '{}_script.err' | gzip -1 > '{}_script.txt.gz'".format(
            self.data, fields, self.out, self.out)
        self.log["script_status"] = subprocess.run(["bash", "-o", "pipefail", "-c", script]).returncode
        if MODE == "ibs":
            ops = "perf script -i '{}' -F {} --hide-call-graph --no-inline 2> '{}_ops.err' | gzip -1 > '{}_ops.txt.gz'".format(
                self.data, IBS_OPS_FIELDS, self.out, self.out)
            self.log["ops_status"] = subprocess.run(["bash", "-o", "pipefail", "-c", ops]).returncode
        self.log["script_seconds"] = time.time() - start
        self.log["data_bytes"] = os.path.getsize(self.data) if os.path.exists(self.data) else None
        self._write_log()


def write_counts(recorder, path):
    """Counts mode: this rank's units' events as the recorder (memory, from the start of the measured simulation) holds
    them, written as each sender's id and its number of events (event_counts_<rank>.npz). Never raises: a failure is
    written into the file instead, and the run goes on."""
    try:
        import numpy as np
        events = recorder.get("events")
        ids, counts = np.unique(np.asarray(events["senders"]), return_counts=True)
        np.savez_compressed(path, ids=ids, counts=counts, events=int(counts.sum()))
    except Exception as err:  # noqa: BLE001
        with open(path + ".error.txt", "w") as f:
            f.write("{!r}\n".format(err))


def write_pages(path):
    """The rank's memory totals (/proc/self/smaps_rollup), among them how much of its anonymous memory sits on large
    pages (AnonHugePages), then the number of jemalloc mappings and the node's transparent huge page settings: the
    candidates' record (delivery_v1_candidates.cpp, record_pages) plus the settings. Never raises: a failure is
    written into the file, and the rank simulates as usual."""
    lines = []
    try:
        with open("/proc/self/smaps_rollup") as f:
            lines.append(f.read().rstrip("\n"))
        with open("/proc/self/maps") as f:
            lines.append("jemalloc mappings: {}".format(sum("libjemalloc" in line for line in f)))
        for name in ("enabled", "defrag", "khugepaged/defrag"):
            with open("/sys/kernel/mm/transparent_hugepage/" + name) as f:
                lines.append("thp {}: {}".format(name, f.read().strip()))
    except Exception as err:  # noqa: BLE001
        lines.append("error: {!r}".format(err))
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
