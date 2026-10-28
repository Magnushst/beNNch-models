"""perf record on ranks 0 and 1 for the measured simulation only.

The recording attaches to the rank's own process with its events off, switches them on just before nest.Run(T) and
off just after, through perf's control fifos, then is written out on the node by perf script (each sample's thread,
event, period and call stack), compressed, into the run's recordings folder. The raw recording stays in
/work/scratch/$USER/nest_perf/<job> (the run's folder if that does not exist). A rank without perf, or whose perf
fails, records nothing and simulates as usual; every step is logged in perf_<rank>.json beside the write-out.
The options and events were tried on a Hambach Zen 3 node before this run (Magnus Stromme, 27 Oct 2026).
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
# No sampling rate: each event has a fixed period, so the periods of its samples add up to its count. 512 kB of buffer
# per thread; 4 kB of each thread's stack per sample, for its call stack; events off until switched on.
OPTIONS = ["-m", "128", "--call-graph=dwarf,4096", "-D", "-1"]
# For each question the first name this node's perf accepts, with a period of roughly a tenth of a second of a thread's
# events in this model: cycles, instructions, loads dispatched, fills from local memory, page walks. Five, so none
# shares a counter.
ROLES = ((("cycles",), 300000000), (("instructions",), 100000000), (("ls_dispatch.ld_dispatch",), 50000000),
         (("ls_refills_from_sys.ls_mabresp_lcl_dram", "ls_dmnd_fills_from_sys.mem_io_local", "ls_any_fills_from_sys.mem_io_local"), 1000000),
         (("ls_tablewalker.dside", "ls_l1_d_tlb_miss.all_l2_miss", "dTLB-load-misses"), 20000))
FIELDS = "tid,event,period,ip,sym,dso"


def accepted(event):
    try:
        out = subprocess.run(["perf", "stat", "-x,", "-e", event + ":u", "--", "true"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.split(",")[0].isdigit() for line in out.stderr.splitlines())


def with_period(event, period, probe):
    """The event with its period in the first form this perf accepts, or None."""
    for form in ("{}/period={}/u".format(event, period), "cpu/{},period={}/u".format(event, period)):
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
        self.rank, self.perf, self.log = rank, None, {"rank": rank, "host": socket.gethostname(), "pid": os.getpid()}
        self.out = os.path.join(recordings, "perf_{}".format(rank))
        scratch = os.path.join("/work/scratch", os.environ.get("USER", ""), "nest_perf", os.environ.get("SLURM_JOB_ID", "nojob"))
        try:
            os.makedirs(scratch, exist_ok=True)
        except OSError:
            scratch = recordings
        self.data = os.path.join(scratch, "rank_{}_perf.data".format(rank))

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
        if self.rank not in RANKS:
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
        names = [next((e for e in role if accepted(e)), None) for role, _ in ROLES]
        events = [with_period(e, period, self.data + ".probe") if e else None for e, (_, period) in zip(names, ROLES)]
        try:
            version = subprocess.run(["perf", "--version"], capture_output=True, text=True).stdout.strip()
        except OSError:
            version = "perf missing"
        self.log.update(events=events, options=OPTIONS, affinity=sorted(os.sched_getaffinity(0)), perf_version=version)
        if None in events:
            self.log["started"] = False
            self._write_log()
            return self
        ctl, ack = self.data + ".ctl", self.data + ".ack"
        for fifo in (ctl, ack):
            if os.path.exists(fifo):
                os.remove(fifo)
            os.mkfifo(fifo)
        # both ends held open here, so neither side ever reads an end of file
        self.ctl, self.ack = os.open(ctl, os.O_RDWR), os.open(ack, os.O_RDWR)
        with open(self.out + "_record.log", "w") as err:
            self.perf = subprocess.Popen(["perf", "record", *OPTIONS, "-e", ",".join(events), "--control",
                                          "fifo:{},{}".format(ctl, ack), "-p", str(os.getpid()), "-o", self.data],
                                         stdout=err, stderr=subprocess.STDOUT)
        try:
            self.log["perf_affinity"] = sorted(os.sched_getaffinity(self.perf.pid))
        except OSError:
            self.log["perf_affinity"] = None
        self.log["started"] = self._command("enable", 300)
        self.log["enabled_at"] = time.time()
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
        script = "perf script -i '{}' -F {} --no-inline 2> '{}_script.err' | gzip -1 > '{}_script.txt.gz'".format(
            self.data, FIELDS, self.out, self.out)
        self.log["script_status"] = subprocess.run(["bash", "-o", "pipefail", "-c", script]).returncode
        self.log["script_seconds"] = time.time() - start
        self.log["data_bytes"] = os.path.getsize(self.data) if os.path.exists(self.data) else None
        self._write_log()


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
